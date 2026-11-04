from envs.env import *
from utils.helper import *
from agents.BaseAgent import *
from components.replay import *
from components.network import *
from components.normalizer import *
import components.exploration


class VanillaDQN(BaseAgent):
  '''
  Implementation of Vanilla DQN with only replay buffer (no target network)
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.cfg = cfg
    self.env_name = cfg['env']['name']
    self.agent_name = cfg['agent']['name']
    self.env = {
      'Train': make_env(cfg['env']['name'], max_episode_steps=int(cfg['env']['max_episode_steps'])),
      'Test': make_env(cfg['env']['name'], max_episode_steps=int(cfg['env']['max_episode_steps']))
    }
    if cfg['env']['name'] in ['NChain-v1', 'LockBernoulli-v0', 'LockGaussian-v0'] and 'cfg' in cfg['env'].keys():
      self.env['Train'].init(**cfg['env']['cfg'])
      self.env['Test'].init(**cfg['env']['cfg'])
    self.config_idx = cfg['config_idx']
    self.device = cfg['device']
    self.discount = cfg['discount']
    self.train_steps = int(cfg['train_steps'])
    self.test_per_episodes = int(cfg['test_per_episodes'])
    self.display_interval = cfg['display_interval']
    self.gradient_clip = cfg['gradient_clip']
    self.action_size = self.get_action_size()
    self.state_size = self.get_state_size()
    self.rolling_score_window = cfg['rolling_score_window']
    self.show_tb = cfg['show_tb']
    self.log_path = {'Train': self.cfg['train_log_path'], 'Test': self.cfg['test_log_path']}
    # Set evaluation (验证) settings: 与 main_atari.py 的 eval_policy / eval_freq 保持一致
    #   eval_freq     : 每隔多少个训练步做一次验证, 0 表示不做验证
    #   eval_episodes : 每次验证跑多少个回合
    #   eval_epsilon  : 验证时的探索率, 0 表示完全贪心
    self.eval_freq = int(cfg.get('eval_freq', 0) or 0)
    self.eval_episodes = int(cfg.get('eval_episodes', 10) or 10)
    self.eval_epsilon = float(cfg.get('eval_epsilon', 0.0) or 0.0)
    self.eval_log_path = cfg.get('eval_log_path', None)
    self.evaluations = []
    # Set tensorboard
    if self.show_tb: self.logger.init_writer()
    # Set normalizers
    # if cfg['env']['input_type'] == 'pixel':
    #   if 'MinAtar' in self.env_name:
        
    self.state_normalizer = RescaleNormalizer()
    self.reward_normalizer = RescaleNormalizer()
      # else:
      #   self.state_normalizer = ImageNormalizer()
      #   self.reward_normalizer = SignNormalizer()
    # elif cfg['env']['input_type'] == 'feature':
    #   self.state_normalizer = RescaleNormalizer()
    #   self.reward_normalizer = RescaleNormalizer()
    # else:
    #   raise ValueError(f"{cfg['env']['input_type']} is not supported.")
    # Create Q value network
    self.Q_net = [None]
    self.Q_net[0] = self.createNN(cfg['env']['input_type']).to(self.device)
    # Set the index of Q_net to be udpated
    self.update_Q_net_index = 0
    # Set optimizer
    self.optimizer = [None]
    self.optimizer[0] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[0].parameters(), **cfg['optimizer']['kwargs'])
    # Set exploration strategy
    epsilon = {
      'steps': float(cfg['epsilon_steps']),
      'start': cfg['epsilon_start'],
      'end': cfg['epsilon_end'],
      'decay': cfg['epsilon_decay']
    }
    # Set exploration strategy
    self.exploration = getattr(components.exploration, cfg['exploration_type'])(cfg['exploration_steps'], epsilon)
    # Set loss function
    self.loss = getattr(torch.nn, cfg['loss'])(reduction='mean')
    # Set replay buffer
    self.replay = getattr(components.replay, cfg['memory_type'])(cfg['memory_size'], keys=['state', 'action', 'next_state', 'reward', 'mask'])
    # Set log dict
    for key in ['state', 'next_state', 'action', 'reward', 'done', 'episode_return', 'episode_step_count']:
      setattr(self, key, {'Train': None, 'Test': None})

  def createNN(self, input_type):
    # Set feature network
    # if input_type == 'pixel':
    layer_dims = [self.cfg['feature_dim']] + self.cfg['hidden_layers'] + [self.action_size]
    # if 'MinAtar' in self.env_name:
    feature_net = Conv2d_MinAtar(in_channels=self.env['Train'].game.state_shape()[2], feature_dim=layer_dims[0])
    #   else:
    #     feature_net = Conv2d_Atari(in_channels=4, feature_dim=layer_dims[0])
    # elif input_type == 'feature':
    #   layer_dims = [self.state_size] + self.cfg['hidden_layers'] + [self.action_size]
    #   feature_net = nn.Identity()
    # Set value network
    assert self.action_type == 'DISCRETE', f'{self.agent_name} only supports discrete action spaces.'
    value_net = MLPCritic(layer_dims=layer_dims, hidden_act=self.cfg['hidden_act'], output_act=self.cfg['output_act'], last_w_scale=1.0)
    NN = DQNNet(feature_net, value_net)
    return NN

  def reset_game(self, mode):
    # Reset the game before a new episode
    self.original_state = self.env[mode].reset() # state before processed
    self.state[mode] = self.state_normalizer(self.original_state)
    self.next_state[mode] = None
    self.action[mode] = None
    self.reward[mode] = None
    self.done[mode] = False
    self.episode_return[mode] = 0
    self.episode_step_count[mode] = 0

  def run_steps(self, render=False):
    # Run for multiple episodes
    self.step_count = 0
    self.episode_count = 0
    self.result = {'Train': [], 'Test': []}
    self.episode_return_list = {'Train': [], 'Test': []}
    self.evaluations = []
    mode = 'Train'
    self.start_time = time.time()
    self.reset_game('Train')
    self.reset_game('Test')
    # Evaluate the untrained policy (与 main_atari.py 的 "Evaluate untrained policy" 一致)
    next_eval_step = 0
    if self.eval_freq > 0:
      self.evaluate()
      next_eval_step = self.eval_freq
    while self.step_count < self.train_steps:
      if mode == 'Train' and self.test_per_episodes > 0 and self.episode_count % self.test_per_episodes == 0:
        mode = 'Test'
      else:
        mode = 'Train'
      # Set Q network to training/evaluation mode
      self.set_net_mode(mode)
      # Run for one episode
      self.run_episode(mode, render)
      # Evaluate the policy every `eval_freq` steps
      if self.eval_freq > 0:
        while self.step_count >= next_eval_step:
          self.evaluate()
          next_eval_step += self.eval_freq
    # Final evaluation after training
    if self.eval_freq > 0 and self.evaluations[-1]['Step'] != self.step_count:
      self.evaluate()

  def run_episode(self, mode, render):
    while not self.done[mode]:
      self.action[mode] = self.get_action(mode)
      if render:
        self.env[mode].render()
      # Take a step
      next_state, self.reward[mode], self.done[mode], _ = self.env[mode].step(self.action[mode])
      self.next_state[mode] = self.state_normalizer(next_state)
      self.reward[mode] = self.reward_normalizer(self.reward[mode])
      self.episode_return[mode] += self.reward[mode]
      self.episode_step_count[mode] += 1
      if mode == 'Train':
        # Save experience
        self.save_experience()
        # Update policy
        if self.time_to_learn():
          self.learn()
        # Update target Q network: used only in DQN variants
        self.update_target_net()
        self.step_count += 1
      # Update state
      self.state[mode] = self.next_state[mode]
      self.original_state = next_state
    # End of one episode
    self.save_episode_result(mode)
    # Reset environment
    self.reset_game(mode)
    if mode == 'Train':
      self.episode_count += 1

  def save_episode_result(self, mode):
    self.episode_return_list[mode].append(self.episode_return[mode])
    rolling_score = np.mean(self.episode_return_list[mode][-1 * self.rolling_score_window[mode]:])
    result_dict = {'Env': self.env_name,
                   'Agent': self.agent_name,
                   'Episode': self.episode_count, 
                   'Step': self.step_count, 
                   'Return': self.episode_return[mode],
                   'Average Return': rolling_score}
    self.result[mode].append(result_dict)
    if self.show_tb:
      self.logger.add_scalar(f'{mode}_Return', self.episode_return[mode], self.step_count)
      self.logger.add_scalar(f'{mode}_Average_Return', rolling_score, self.step_count)
    if mode == 'Test' or self.episode_count % self.display_interval == 0 or self.step_count >= self.train_steps:
      # Save result to files
      result = pd.DataFrame(self.result[mode])
      result['Env'] = result['Env'].astype('category')
      result['Agent'] = result['Agent'].astype('category')
      result.to_feather(self.log_path[mode])
      # Show log
      speed = self.step_count / (time.time() - self.start_time)
      eta = (self.train_steps - self.step_count) / speed / 60 if speed>0 else -1
      self.logger.info(f'<{self.config_idx}> [{mode}] Episode {self.episode_count}, Step {self.step_count}: Average Return({self.rolling_score_window[mode]})={rolling_score:.2f}, Return={self.episode_return[mode]:.2f}, Speed={speed:.2f} (steps/s), ETA={eta:.2f} (mins)')

  def get_eval_env(self):
    '''
    Lazily create a separated environment used for evaluation (验证).
    与 main_atari.py 中 `eval_env = make_atari(..., seed=seed+100)` 的做法一致:
    验证使用独立的环境, 不会干扰训练/Test 环境的状态与随机数。
    '''
    if 'Eval' not in self.env:
      self.env['Eval'] = make_env(self.env_name, max_episode_steps=int(self.cfg['env']['max_episode_steps']))
      seed = int(self.cfg['seed']) + 100
      try:
        self.env['Eval'].seed(seed)
        self.env['Eval'].action_space.np_random.seed(seed)
      except Exception:
        pass
      if self.env_name in ['NChain-v1', 'LockBernoulli-v0', 'LockGaussian-v0'] and 'cfg' in self.cfg['env'].keys():
        self.env['Eval'].init(**self.cfg['env']['cfg'])
    return self.env['Eval']

  def evaluate(self):
    '''
    Run `eval_episodes` episodes with a (ε-)greedy policy and record the result.
    对应 main_atari.py 的 `eval_policy()`。
    Return: 本次验证的平均回合回报
    '''
    mode = 'Eval'
    env = self.get_eval_env()
    # Set Q network to evaluation mode (e.g. disable NoisyNet noise)
    self.set_net_mode('Test')
    returns, steps = [], []
    for _ in range(self.eval_episodes):
      self.reset_game(mode)
      while not self.done[mode]:
        state = to_tensor(self.state[mode], device=self.device).unsqueeze(0)
        q_values = self.get_action_selection_q_values(state)
        if self.eval_epsilon > 0 and np.random.rand() < self.eval_epsilon:
          action = np.random.randint(self.action_size)
        else:
          action = int(np.argmax(q_values))
        # Take a step
        next_state, self.reward[mode], self.done[mode], _ = env.step(action)
        self.next_state[mode] = self.state_normalizer(next_state)
        self.reward[mode] = self.reward_normalizer(self.reward[mode])
        self.episode_return[mode] += self.reward[mode]
        self.episode_step_count[mode] += 1
        # Update state
        self.state[mode] = self.next_state[mode]
      returns.append(float(self.episode_return[mode]))
      steps.append(int(self.episode_step_count[mode]))
    # Set Q network back to training mode
    self.set_net_mode('Train')
    return self.save_eval_result(returns, steps)

  def save_eval_result(self, returns, steps):
    '''
    Print and save the evaluation (验证) result, 对应 main_atari.py 中:
        evaluations.append(avg_reward); np.save(result_dir + file_name, evaluations)

    保存的文件:
      * evaluations.npy : 每次验证的平均回报 (与 main_atari.py 完全一致的格式)
      * evaluations.csv : 每次验证的详细统计 (Step / 均值 / 标准差 / 最小 / 最大)
    '''
    returns = np.asarray(returns, dtype=np.float64)
    step_count = getattr(self, 'step_count', 0)
    episode_count = getattr(self, 'episode_count', 0)
    avg_return = float(returns.mean())
    result_dict = {'Env': self.env_name,
                   'Agent': self.agent_name,
                   'Episode': episode_count,
                   'Step': step_count,
                   'Average Return': avg_return,
                   'Std Return': float(returns.std()),
                   'Min Return': float(returns.min()),
                   'Max Return': float(returns.max()),
                   'Average Step': float(np.mean(steps))}
    self.evaluations.append(result_dict)
    # Show log
    start_time = getattr(self, 'start_time', None)
    speed = step_count / (time.time() - start_time) if start_time else -1
    eta = (self.train_steps - step_count) / speed / 60 if speed > 0 else -1
    log_str = (f'<{self.config_idx}> [Eval] Episode {episode_count}, Step {step_count}: '
               f'Average Return={avg_return:.2f}, Std={result_dict["Std Return"]:.2f}, '
               f'Min={result_dict["Min Return"]:.2f}, Max={result_dict["Max Return"]:.2f}, '
               f'Episodes={len(returns)}, Speed={speed:.2f} (steps/s), ETA={eta:.2f} (mins)')
    self.logger.info(log_str)
    print(log_str)  # 同时打印到标准输出, 方便直接观察验证结果
    if self.show_tb:
      self.logger.add_scalar('Eval_Average_Return', avg_return, step_count)
    # Save result to files
    if self.cfg.get('eval_log_path', None):
      eval_log_path = self.cfg['eval_log_path']
      make_dir(os.path.dirname(eval_log_path))
      np.save(eval_log_path, np.array([e['Average Return'] for e in self.evaluations]))
      pd.DataFrame(self.evaluations).to_csv(eval_log_path.replace('.npy', '.csv'), index=False)
    # Save model (optional)
    if self.cfg.get('save_model', False):
      self.save_model(self.cfg['model_path'].replace('.pt', f'_{step_count}.pt'))
    return avg_return

  def get_action(self, mode='Train'):
    '''
    Uses the local Q network and an epsilon greedy policy to pick an action
    PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
    a "fake" dimension to make it a mini-batch rather than a single observation
    '''
    state = to_tensor(self.state[mode], device=self.device)
    state = state.unsqueeze(0) # Add a batch dimension (Batch, Channel, Height, Width)
    q_values = self.get_action_selection_q_values(state)
    if mode == 'Test':
      action = np.argmax(q_values) # During test, select best action
    elif mode == 'Train':
      action = self.exploration.select_action(q_values, self.step_count)
    return action

  def time_to_learn(self):
    """
    Return boolean to indicate whether it is time to learn:
    - The agent is not on exploration stage
    - There are enough experiences in replay buffer
    """
    if self.step_count > self.cfg['exploration_steps'] and self.step_count % self.cfg['network_update_steps'] == 0:
      return True
    else:
      return False

  def update_target_net(self):
    pass

  def learn(self):
    mode = 'Train'
    batch = self.replay.sample(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['batch_size'])
    q, q_target = self.compute_q(batch), self.compute_q_target(batch)
    # Compute loss
    loss = self.loss(q, q_target)
    # Take an optimization step
    self.optimizer[self.update_Q_net_index].zero_grad()
    loss.backward()
    if self.gradient_clip > 0:
      nn.utils.clip_grad_norm_(self.Q_net[self.update_Q_net_index].parameters(), self.gradient_clip)
    self.optimizer[self.update_Q_net_index].step()
    if self.show_tb:
      self.logger.add_scalar(f'Loss', loss.item(), self.step_count)

  def compute_q_target(self, batch):
    with torch.no_grad():
      q_next = self.Q_net[0](batch.next_state).max(1)[0]
      q_target = batch.reward + self.discount * q_next * batch.mask
    return q_target
  
  def compute_q(self, batch):
    # Convert actions to long so they can be used as indexes
    action = batch.action.long().unsqueeze(1)
    q = self.Q_net[self.update_Q_net_index](batch.state).gather(1, action).squeeze()
    return q

  def save_experience(self):
    mode = 'Train'
    prediction = {}
    if self.reward[mode] is not None:
      prediction['state'] = to_tensor(self.state[mode], self.device)
      prediction['action'] = to_tensor(self.action[mode], self.device)
      prediction['next_state'] = to_tensor(self.next_state[mode], self.device)
      prediction['mask'] = to_tensor(1-self.done[mode], self.device)
      prediction['reward'] = to_tensor(self.reward[mode], self.device)
    self.replay.add(prediction)

  def get_action_size(self):
    mode = 'Train'
    if isinstance(self.env[mode].action_space, Discrete):
      self.action_type = 'DISCRETE'
      return self.env[mode].action_space.n
    elif isinstance(self.env[mode].action_space, Box):
      self.action_type = 'CONTINUOUS'
      # Set the minimum/maximum/limit values of the action spaces
      self.action_min = min(self.env[mode].action_space.low)
      self.action_max = max(self.env[mode].action_space.high)
      self.action_lim = max(abs(self.action_min), self.action_max)
      return self.env[mode].action_space.shape[0]
    else:
      raise ValueError('Unknown action type.')

  def get_state_size(self):
    mode = 'Train'
    if isinstance(self.env[mode].observation_space, Discrete):
      if hasattr(self.env[mode], 'env'):
        return self.env[mode].env.observation_space.n
      else:
        return self.env[mode].observation_space.n
    else: # Box, MultiBinary
      if hasattr(self.env[mode], 'env'):
        return int(np.prod(self.env[mode].env.observation_space.shape))
      else:
        return int(np.prod(self.env[mode].observation_space.shape))

  def set_net_mode(self, mode):
    if mode == 'Test':
      for i in range(len(self.Q_net)):
        self.Q_net[i].eval() # Set Q network to evaluation mode
    elif mode == 'Train':
      for i in range(len(self.Q_net)):
        self.Q_net[i].train() # Set Q network back to training mode
  
  def get_action_selection_q_values(self, state):
    q_values = self.Q_net[0](state)
    q_values = to_numpy(q_values).flatten()
    return q_values

  def save_model(self, model_path):
    state_dicts = {} 
    for i in range(len(self.Q_net)):
      state_dicts[i] = self.Q_net[i].state_dict()
    torch.save(state_dicts, model_path)
  
  def load_model(self, model_path):
    state_dicts = torch.load(model_path)
    for i in range(len(self.Q_net)):
      self.Q_net[i].load_state_dict(state_dicts[i])
      self.Q_net[i] = self.Q_net[i].to(self.device)