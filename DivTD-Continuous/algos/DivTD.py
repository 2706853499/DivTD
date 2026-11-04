import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

# 表示DivQ与当前的TD7结合进行实验
@dataclass
class Hyperparameters:
    # Generic
    batch_size: int = 256
    buffer_size: int = 1e6
    discount: float = 0.99
    target_update_rate: int = 250
    exploration_noise: float = 0.1
    
    # TD3
    target_policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    
    # LAP
    alpha: float = 0.4
    min_priority: float = 1
    
    # TD3+BC
    lmbda: float = 0.1
    
    # Checkpointing
    max_eps_when_checkpointing: int = 20
    steps_before_checkpointing: int = 75e4 
    reset_weight: float = 0.9
    
    # Encoder Model
    zs_dim: int = 256
    enc_hdim: int = 256
    enc_activ: Callable = F.elu
    encoder_lr: float = 3e-4
    
    # Critic Model
    critic_hdim: int = 256
    critic_activ: Callable = F.elu
    critic_lr: float = 3e-4
    
    critic_quantile_diversity_coeff: float = 1e-4
    critic_quantile_diversity_margin: float = 4e-2
    
    # Actor Model
    actor_hdim: int = 256
    actor_activ: Callable = F.relu
    actor_lr: float = 3e-4


def AvgL1Norm(x, eps=1e-8):
    return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)


def LAP_huber(x, min_priority=1):
    return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).sum(1).mean()


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def soft_clamp(x, _min=None, _max=None):
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


 
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.relu):
        super(Actor, self).__init__()

        self.activ = activ

        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)
        

    def forward(self, state, zs):
        a = AvgL1Norm(self.l0(state))
        a = torch.cat([a, zs], 1)
        a = self.activ(self.l1(a))
        a = self.activ(self.l2(a))
        return torch.tanh(self.l3(a))


class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
        super(Encoder, self).__init__()

        self.activ = activ

        # state encoder
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)
        
        # state-action encoder
        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)
    

    def zs(self, state):
        zs = self.activ(self.zs1(state))
        zs = self.activ(self.zs2(zs))
        zs = AvgL1Norm(self.zs3(zs))
        return zs


    def zsa(self, zs, action):
        zsa = self.activ(self.zsa1(torch.cat([zs, action], 1)))
        zsa = self.activ(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        return zsa


 
class QuantileCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, n_quantiles=25, activ=F.elu):
        super().__init__()
        self.activ = activ
        self.q0 = nn.Linear(state_dim + action_dim, hdim)
        
        self.q1 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.dop1 =  nn.Dropout1d(0.1) #nn.LayerNorm(hdim)#
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, n_quantiles)

    def forward(self, state, action, zsa, zs):
        sa = torch.cat([state, action], 1)
        embeddings = torch.cat([zsa, zs], 1)
        q = AvgL1Norm(self.q0(sa))
        q = torch.cat([q, embeddings], 1)
        q = self.activ(self.dop1(self.q1(q)))
        q = self.activ(self.q2(q))
        return self.q3(q)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu, n_quantiles=25, n_nets=5):
        super(Critic, self).__init__()
        self.n_quantiles = n_quantiles
        self.n_nets = n_nets
        self.nets = nn.ModuleList(
            [
                QuantileCriticNet(
                    state_dim,
                    action_dim,
                    zs_dim=zs_dim,
                    hdim=hdim,
                    n_quantiles=n_quantiles,
                    activ=activ,
                )
                for _ in range(n_nets)
            ]
        )

    def forward(self, state, action, zsa, zs):
        quantiles = [net(state, action, zsa, zs).unsqueeze(1) for net in self.nets]
        return torch.cat(quantiles, dim=1)
  
class DivTD(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        lamda=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        n_quantiles=25,
        n_nets=5,
        top_quantiles_to_drop_per_net=2,
        device="cuda:0",
        offline=False,
        N=5,M=4,alpha=0.1,
        hp=Hyperparameters(),
        **kwargs,
    ): 
        # Changing hyperparameters example: hp=Hyperparameters(batch_size=128)
        
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.hp = hp
        self.hp.discount = discount
        self.hp.target_policy_noise = policy_noise
        self.hp.noise_clip = noise_clip
        self.n_quantiles = int(n_quantiles)
        self.n_nets = int(n_nets)
        self.top_quantiles_to_drop_per_net = int(top_quantiles_to_drop_per_net)
        self.total_quantiles = self.n_quantiles * self.n_nets
        self.total_quantiles_to_drop = self.top_quantiles_to_drop_per_net * self.n_nets
        self.quantile_tau = (
            (torch.arange(self.n_quantiles, device=self.device, dtype=torch.float32) + 0.5)
            / float(self.n_quantiles)
        ).view(1, 1, self.n_quantiles, 1)
        
       
       

        self.actor = Actor(state_dim, action_dim, hp.zs_dim, hp.actor_hdim, hp.actor_activ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr)
        self.actor_target = copy.deepcopy(self.actor)
        self.N=N
        self.M=M
        self.alpha=alpha

        self.critic = Critic(
            state_dim,
            action_dim,
            hp.zs_dim,
            hp.critic_hdim,
            hp.critic_activ,
            n_quantiles=1,
            n_nets=self.N,
        ).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr)
        self.critic_target = copy.deepcopy(self.critic)

        self.encoder = Encoder(state_dim, action_dim, hp.zs_dim, hp.enc_hdim, hp.enc_activ).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=hp.encoder_lr)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)
 

        self.checkpoint_actor = copy.deepcopy(self.actor)
        self.checkpoint_encoder = copy.deepcopy(self.encoder)

        self.replay_buffer = utils.ReplayBuffer(
            state_dim,
            action_dim,
            max_size=hp.buffer_size,
            device=self.device,
            batch_size=hp.batch_size,
            max_action=max_action,
            normalize_actions=True,
           
        )
       

        self.max_action = max_action
        self.offline = offline

        self.training_steps = 0
        # Checkpointing tracked values
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8

        # Value clipping tracked values
        self.max = -1e8
        self.min = 1e8
        self.max_target = 0
        self.min_target = 0
        
        self.list_index = []

    def _sample_array_batch(self, replay_buffer, batch_size):
        indices = np.random.randint(0, replay_buffer.size, size=int(batch_size))
        state = torch.tensor(replay_buffer.state[indices], dtype=torch.float32, device=self.device)
        action = torch.tensor(replay_buffer.action[indices], dtype=torch.float32, device=self.device)
        next_state = torch.tensor(replay_buffer.next_state[indices], dtype=torch.float32, device=self.device)
        reward = torch.tensor(replay_buffer.reward[indices], dtype=torch.float32, device=self.device)
        not_done = torch.tensor(replay_buffer.not_done[indices], dtype=torch.float32, device=self.device)
        return state, action, next_state, reward, not_done
 
 
   
    def _quantile_huber_loss(self, quantiles: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
      
        pairwise_delta = samples - quantiles  
        # negative_mask = pairwise_delta < 0
        abs_pairwise_delta = pairwise_delta.abs()
        huber_loss = torch.where(
            abs_pairwise_delta > 1.0,
            abs_pairwise_delta - 0.5,
            0.5 * abs_pairwise_delta.pow(2),
        )
       
        return huber_loss.mean()  
 
 
    def select_action(self, state, deterministic=False, use_checkpoint=False ):
        with torch.no_grad():
            state = torch.tensor(state.reshape(1,-1), dtype=torch.float, device=self.device)
            if use_checkpoint: 
                zs = self.checkpoint_encoder.zs(state)
                action = self.checkpoint_actor(state, zs) 
            else: 
                zs = self.fixed_encoder.zs(state)
                action = self.actor(state, zs) 
            return   (action  * self.max_action).cpu().data.numpy().flatten() 

 
    def train(self, replay_buffer=None, batch_size=256):
        self.training_steps += 1
        if replay_buffer is None:
            replay_buffer = self.replay_buffer
       
        state, action, next_state, reward, not_done = self._sample_array_batch(
            replay_buffer,
            batch_size,
        )
       
      
        #########################
        # Update Encoder
        #########################
        with torch.no_grad():
            next_zs = self.encoder.zs(next_state)

        zs = self.encoder.zs(state)
        pred_zs = self.encoder.zsa(zs, action)
        encoder_loss = F.mse_loss(pred_zs, next_zs) 
        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()


        # rand_inx = np.random.randint(0,self.n_nets)
        # current_batch_size = state.shape[0]
        q_indices = np.random.choice(
                self.N,
                size=self.M,
                replace=False,
        )
        self.list_index.extend(list(q_indices))
        with torch.no_grad():
             
            fixed_target_zs = self.fixed_encoder_target.zs(next_state)
            noise = (torch.randn_like(action) * self.hp.target_policy_noise).clamp(-self.hp.noise_clip, self.hp.noise_clip)
            next_action = (self.actor_target(next_state, fixed_target_zs) + noise).clamp(-1,1)
            
            fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)
            next_z = self.critic_target(next_state, next_action, fixed_target_zsa, fixed_target_zs)
            sorted_z =  next_z.view(batch_size, -1) 
            if self.M!=1:
                target_std = not_done *self.hp.discount *torch.std(sorted_z,dim=1,keepdim=True)
             
            Q_target = reward + not_done * self.hp.discount * sorted_z.mean(axis=1,keepdim=True)
            
 
            train_fixed_zs = self.fixed_encoder.zs(state)
            train_fixed_zsa = self.fixed_encoder.zsa(train_fixed_zs, action)
           

        Q = self.critic(state, action, train_fixed_zsa, train_fixed_zs) [:,q_indices,:].view(batch_size, -1)
        critic_loss = self._quantile_huber_loss(Q, Q_target)
        if self.M!=1:
            cur_std = torch.std(Q,dim=1)
            critic_loss+=self.alpha*F.mse_loss(cur_std,target_std)
        

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        #########################
        # Update Actor
        #########################
        if self.training_steps % self.hp.policy_freq == 0:
           
            indx = list(set(list(self.list_index)) )

            actor_zs = self.fixed_encoder.zs(state)
            new_action = self.actor(state, actor_zs).clamp(-1, 1)
            actor_zsa = self.fixed_encoder.zsa(actor_zs, new_action)
            Qs = self.critic(state, new_action, actor_zsa, actor_zs) [:,indx,:]
            q_guided_loss = -Qs.mean()
            self.actor_optimizer.zero_grad()
            q_guided_loss.backward()
            self.actor_optimizer.step()

        #########################
        # Update Iteration
        #########################
        if self.training_steps % self.hp.target_update_rate == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())
            
            self.max_target = self.max
            self.min_target = self.min
      


    # If using checkpoints: run when each episode terminates
    def maybe_train_and_checkpoint(self, ep_timesteps, ep_return):
        self.eps_since_update += 1
        self.timesteps_since_update += ep_timesteps

        self.min_return = min(self.min_return, ep_return)

        # End evaluation of current policy early
        if self.min_return < self.best_min_return:
            self.train_and_reset()

        # Update checkpoint
        elif self.eps_since_update == self.max_eps_before_update:
            self.best_min_return = self.min_return
            self.checkpoint_actor.load_state_dict(self.actor.state_dict())
            self.checkpoint_encoder.load_state_dict(self.fixed_encoder.state_dict())
            
            self.train_and_reset()


    # Batch training
    def train_and_reset(self):
        for _ in range(self.timesteps_since_update):
            if self.training_steps == self.hp.steps_before_checkpointing:
                self.best_min_return *= self.hp.reset_weight
                self.max_eps_before_update = self.hp.max_eps_when_checkpointing
            
            self.train()

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.encoder.state_dict(), filename + "_encoder")
        torch.save(self.encoder_optimizer.state_dict(), filename + "_encoder_optimizer")
       

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)
        try:
            self.encoder.load_state_dict(torch.load(filename + "_encoder", map_location=self.device))
            self.encoder_optimizer.load_state_dict(torch.load(filename + "_encoder_optimizer", map_location=self.device))
            self.fixed_encoder = copy.deepcopy(self.encoder)
            self.fixed_encoder_target = copy.deepcopy(self.encoder)
        except FileNotFoundError:
            pass
    
         

 
 
 
