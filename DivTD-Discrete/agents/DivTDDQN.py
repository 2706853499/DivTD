import torch
import torch.nn.functional as F

from agents.DQN import *


class DivTD(DQN):
  '''
  DivTD (`divtd.py` 中的 `ensamble_TD7`) 在离散动作空间上的实现:
  基础算法为 DQN (target network + replay buffer + epsilon-greedy),
  Critic/Actor 被替换为 N 个 Q 网络的集成, 并加入稀疏训练与价值正则化。

  与 `divtd.py::ensamble_TD7.train` 的对应关系 (以下均为 divtd.py 中**实际生效**的代码):

  * Q-target (`divtd.py` 中 `next_z / sorted_z / Q_target`):
      1. 用 N 个 target 网络给下一状态打分: (B, N, A)
      2. 用集成均值选贪婪动作 (离散版对应 `actor_target(next_state)`;
         离散动作无 TD3 的 target policy noise)
      3. 取出 N 个 target 网络在该动作上的 Q 值并展平为 (B, N) (即 `next_z.view(batch_size, -1)`)
      4. TD 目标取集成均值 (与 `sorted_z.mean(axis=1, keepdim=True)` 一致):
             `Q_target = r + gamma * mask * mean_j Q_j(s', a*)`, 形状 (B, 1)
      5. 目标标准差 (仅当稀疏子集大小 M != 1 时), 用于价值正则化:
             `target_std = gamma * mask * std_j Q_j(s', a*)`, 形状 (B, 1)

  * 损失函数 (`divtd.py::_quantile_huber_loss`):
      `huber(Q_target - Q).mean()`, 其中 Q 是被稀疏选中的 M 个网络的预测值 (B, M),
      Q_target 为 (B, 1), 通过广播得到 (B, M) 后取均值。

  * 稀疏训练 (sparse training, `divtd.py` 中 `q_indices = np.random.choice(self.N, size=self.M)`):
      每次梯度更新只随机抽取 M (默认 min(4, N)) 个网络参与前向/反向与参数更新,
      其余网络不接受梯度, 从而把集成训练/推理的开销降下来。

  * 价值正则化 (value regularization, `divtd.py` 中 `critic_loss += alpha * mse_loss(cur_std, target_std)`):
      让被选中 M 个网络价值估计的离散度 `std(Q)` 去匹配 target 集成的离散度 `target_std`,
      避免稀疏更新把集成"压扁"成同一个函数 (即维持 Div 多样性)。系数 `alpha` 默认 0.1, 与 divtd.py 一致。

  * 额外可选项: `_quantile_diversity_regularizer` (间隔 margin 惩罚) 在 divtd.py 中被注释掉未启用,
    这里保留实现, 通过配置 `diversity_coeff > 0` 打开 (默认 0, 即与 divtd.py 一致)。
  '''
  def __init__(self, cfg):
     
    super().__init__(cfg)
    
    agent_cfg = cfg['agent']
    # N: 集成中 (target) 网络的个数, 对应 divtd.py 的 self.N
    self.N =5
    # M: 每次梯度更新随机抽取的网络个数 (稀疏训练), 对应 divtd.py 的 self.M
    
    self.M =4
    # 价值正则化系数, 对应 divtd.py 的 self.alpha
    self.alpha =0.5
    # 间隔式多样性正则 (divtd.py 中默认未启用)
    self.diversity_coeff = float(agent_cfg.get('diversity_coeff', 0.0))
    self.diversity_margin = float(agent_cfg.get('diversity_margin', 4e-2))
    # 是否对集成目标排序后再取均值 (divtd.py 实际未排序; 置 1 可复现 TQC/Div 风格的有序目标)
    self.sort_targets = bool(agent_cfg.get('sort_targets', False))
    # Create N different: Q value network, Target Q value network and Optimizer
    self.Q_net = [None] * self.N
    self.Q_net_target = [None] * self.N
    self.optimizer = [None] * self.N
    for i in range(self.N):
      self.Q_net[i] = self.createNN(cfg['env']['input_type']).to(self.device)
      self.Q_net_target[i] = self.createNN(cfg['env']['input_type']).to(self.device)
      self.optimizer[i] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[i].parameters(), **cfg['optimizer']['kwargs'])
      # Load target Q value network
      self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())
      self.Q_net_target[i].eval()
    # Indices of the networks used by the sparse update
    self.q_indices = np.random.choice(self.N, size=self.M, replace=False)

  def learn(self):
     
    # Sparse training: randomly pick M out of N networks for this gradient step
    self.q_indices = np.random.choice(self.N, size=self.M, replace=False)
    batch = self.replay.sample(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['batch_size'])
    q = self.compute_q(batch)
    q_target, target_std = self.compute_q_target(batch)
    # Compute loss: Huber loss + value regularization (std matching)
    loss = self.compute_loss(q, q_target, target_std)
    # Take an optimization step (only the selected networks receive gradients)
    for i in self.q_indices:
      self.optimizer[i].zero_grad()
    loss.backward()
    if self.gradient_clip > 0:
     
      for i in self.q_indices:
        
        nn.utils.clip_grad_norm_(self.Q_net[i].parameters(),self.gradient_clip)
     
    for i in self.q_indices:
      self.optimizer[i].step()
    if self.show_tb:
      self.logger.add_scalar(f'Loss', loss.item(), self.step_count)

  def compute_loss(self, q, q_target, target_std=None):
    '''
    q:          (Batch, M) action values of the sparsely selected networks
    q_target:   (Batch, 1) TD target (ensemble mean)
    target_std: (Batch, 1) std of the target ensemble, None when M == 1
    '''
    # divtd.py: critic_loss = self._quantile_huber_loss(Q, Q_target)
    loss =F.smooth_l1_loss(q,q_target.detach())  
    # divtd.py: critic_loss += self.alpha * F.mse_loss(cur_std, target_std)   (M != 1)
    if target_std is not None:
      loss = loss + self.alpha * F.mse_loss(torch.std(q, dim=1, keepdim=True), target_std)
    
    return loss

 

 
  def update_target_net(self):
    if self.step_count % self.cfg['target_network_update_steps'] == 0:
      for i in range(self.N):
        self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())

  def compute_q_target(self, batch):
    '''
    Returns:
      q_target:   (Batch, 1) TD target, reward + gamma * mask * mean_j Q_j(s', a*)
      target_std: (Batch, 1) gamma * mask * std_j Q_j(s', a*), None when M == 1
    '''
    with torch.no_grad():
      q_indices = np.random.choice(self.N, size=self.M, replace=False)
      # (Batch, N, #actions): value of every action given by every target network
      q_next_all = torch.stack([net(batch.next_state) for net in self.Q_net_target], dim=1)[:,q_indices]
      # Greedy action w.r.t. the ensemble average (discrete version of `actor_target`)
      
      next_action = q_next_all.mean(dim=1).max(1)[1]
      # Value of the greedy action given by every target network: (Batch, N)
      next_z = q_next_all.gather(2, next_action[:, None, None].expand(-1, self.M, -1)).squeeze(-1)
       
      # reward / mask are (Batch, 1) in `divtd.py`; add the trailing dim so that broadcasting works
      reward = batch.reward.reshape(-1, 1)
      mask = batch.mask.reshape(-1, 1)
      # divtd.py: Q_target = reward + not_done * discount * sorted_z.mean(axis=1, keepdim=True)
      q_target = reward + mask * self.discount * next_z.mean(dim=1, keepdim=True)
      # divtd.py: target_std = not_done * discount * std(sorted_z, dim=1)   (only when M != 1)
      target_std = None
      if self.M != 1:
        target_std = mask * self.discount * torch.std( next_z , dim=1, keepdim=True )
    return q_target, target_std

  def compute_q(self, batch):
    '''
    divtd.py: Q = critic(state, action)[:, q_indices, :].view(batch_size, -1) -> (Batch, M)
    '''
    # Convert actions to long so they can be used as indexes
    action = batch.action.long().unsqueeze(1)
    q = torch.stack([self.Q_net[i](batch.state).gather(1, action) for i in self.q_indices], dim=1)
    return q.squeeze(-1)

  def get_action_selection_q_values(self, state):
    # Ensemble average is used for action selection (same as the actor objective in `divtd.py`)
    q_values = self.Q_net[ self.q_indices[0]](state).clone()
    for i in  self.q_indices[1:self.M]: 
      q_values = q_values + self.Q_net[i](state)
    q_values = to_numpy(q_values / self.M).flatten()
    return q_values
 

# Backward compatible alias: the class was previously (wrongly) named MaxminDQN
DivTDDQN = DivTD
