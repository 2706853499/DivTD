import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def avg_l1_norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def lap_huber(x, min_priority=1.0):
    return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).sum(1).mean()


class LAP(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        max_size=1e6,
        batch_size=256,
        max_action=1.0,
        normalize_actions=True,
        prioritized=True,
    ):
        max_size = int(max_size)
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device
        self.batch_size = batch_size

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.prioritized = prioritized
        if prioritized:
            self.priority = torch.zeros(max_size, device=device)
            self.max_priority = 1.0

        self.normalize_actions = max_action if normalize_actions else 1.0

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action / self.normalize_actions
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        if self.prioritized:
            self.priority[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size

        if self.prioritized:
            csum = torch.cumsum(self.priority[: self.size], dim=0)
            val = torch.rand(size=(batch_size,), device=self.device) * csum[-1]
            self.ind = torch.searchsorted(csum, val).cpu().data.numpy()
        else:
            self.ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.tensor(self.state[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.action[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.next_state[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.reward[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.not_done[self.ind], dtype=torch.float, device=self.device),
        )

    def update_priority(self, priority):
        if self.prioritized:
            self.priority[self.ind] = priority.reshape(-1).detach()
            self.max_priority = max(float(priority.max()), self.max_priority)

    def reset_max_priority(self):
        if self.prioritized and self.size > 0:
            self.max_priority = float(self.priority[: self.size].max())


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hidden_dim=256, activ=F.relu):
        super().__init__()
        self.activ = activ
        self.l0 = nn.Linear(state_dim, hidden_dim)
        self.l1 = nn.Linear(zs_dim + hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, zs):
        a = avg_l1_norm(self.l0(state))
        a = torch.cat([a, zs], dim=1)
        a = self.activ(self.l1(a))
        a = self.activ(self.l2(a))
        return torch.tanh(self.l3(a))


class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hidden_dim=256, activ=F.elu):
        super().__init__()
        self.activ = activ

        self.zs1 = nn.Linear(state_dim, hidden_dim)
        self.zs2 = nn.Linear(hidden_dim, hidden_dim)
        self.zs3 = nn.Linear(hidden_dim, zs_dim)

        self.zsa1 = nn.Linear(zs_dim + action_dim, hidden_dim)
        self.zsa2 = nn.Linear(hidden_dim, hidden_dim)
        self.zsa3 = nn.Linear(hidden_dim, zs_dim)
        
        # 添加权重初始化以增强数值稳定性
        init_w = 1e-3
        for layer in [self.zs1, self.zs2, self.zs3, self.zsa1, self.zsa2, self.zsa3]:
            layer.weight.data.uniform_(-init_w, init_w)
            layer.bias.data.uniform_(-init_w, init_w)

    def zs(self, state):
        zs = self.activ(self.zs1(state))
        zs = self.activ(self.zs2(zs))
        zs = self.zs3(zs)
         
        return avg_l1_norm(zs)

    def zsa(self, zs, action):
        zsa = self.activ(self.zsa1(torch.cat([zs, action], dim=1)))
        zsa = self.activ(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        
        return zsa


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hidden_dim=256, activ=F.elu):
        super().__init__()
        self.activ = activ

        self.q01 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1 = nn.Linear(2 * zs_dim + hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, hidden_dim)
        self.q3 = nn.Linear(hidden_dim, 1)

        self.q02 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q4 = nn.Linear(2 * zs_dim + hidden_dim, hidden_dim)
        self.q5 = nn.Linear(hidden_dim, hidden_dim)
        self.q6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, zsa, zs):
        sa = torch.cat([state, action], dim=1)
        embeddings = torch.cat([zsa, zs], dim=1)

        q1 = avg_l1_norm(self.q01(sa))
        q1 = torch.cat([q1, embeddings], dim=1)
        q1 = self.activ(self.q1(q1))
        q1 = self.activ(self.q2(q1))
        q1 = self.q3(q1)

        q2 = avg_l1_norm(self.q02(sa))
        q2 = torch.cat([q2, embeddings], dim=1)
        q2 = self.activ(self.q4(q2))
        q2 = self.activ(self.q5(q2))
        q2 = self.q6(q2)
        return q1, q2


class TD7(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        lamda=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        device="cpu",
        hidden_dim=256,
        zs_dim=256,
        target_update_rate=250,
        min_priority=1.0,
        buffer_size=1e6,
        alpha=0.4,
        **kwargs,
    ):
        self.device = torch.device(device)
        self.max_action = max_action
        self.discount = discount
        self.tau = lamda
        self.policy_noise = policy_noise / max_action
        self.noise_clip = noise_clip / max_action
        self.policy_freq = policy_freq
        self.target_update_rate = target_update_rate
        self.min_priority = min_priority
        self.alpha = alpha
        self.total_it = 0
        self.max_target = 0.0
        self.min_target = 0.0
        self.max_q_uptodate = -1e8
        self.min_q_uptodate = 1e8
        self.batch_size = kwargs.get("batch_size", 256)

        self.actor = Actor(state_dim, action_dim, zs_dim=zs_dim, hidden_dim=hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim, zs_dim=zs_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.encoder = Encoder(state_dim, action_dim, zs_dim=zs_dim, hidden_dim=hidden_dim).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=3e-4)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)

        self.replay_buffer = LAP(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
            max_size=buffer_size,
            batch_size=self.batch_size,
            max_action=max_action,
            normalize_actions=True,
            prioritized=True,
        )
        self.list_index = []

    def select_action(self, state,deterministic=False):
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=self.device)
            zs = self.fixed_encoder.zs(state)
            action = self.actor(state, zs)
            return (action.clamp(-1, 1) * self.max_action).cpu().data.numpy().flatten()

    def add(self, state, action, next_state, reward, done):
        self.replay_buffer.add(state, action, next_state, reward, done)

    def train(self, replay_buffer=None, batch_size=256, env=None, sz=None):
        if self.replay_buffer.size < batch_size:
            return

        self.replay_buffer.batch_size = batch_size
        self.total_it += 1
        state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

        normalized_action = action

        with torch.no_grad():
            next_zs_online = self.encoder.zs(next_state)

        zs = self.encoder.zs(state)
        pred_zsa = self.encoder.zsa(zs, normalized_action)
        encoder_loss = F.mse_loss(pred_zsa, next_zs_online)

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

        with torch.no_grad():
            target_zs = self.fixed_encoder_target.zs(next_state)
            noise = (torch.randn_like(normalized_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, target_zs) + noise).clamp(-1, 1)
            target_zsa = self.fixed_encoder_target.zsa(target_zs, next_action)

            target_q1, target_q2 = self.critic_target(next_state, next_action, target_zsa, target_zs)
            target_q = torch.min(target_q1, target_q2)
            
            target_q = reward + not_done * self.discount * target_q.clamp(self.min_target, self.max_target)

            self.max_q_uptodate = max(self.max_q_uptodate, float(target_q.max()))
            self.min_q_uptodate = min(self.min_q_uptodate, float(target_q.min()))

            fixed_zs = self.fixed_encoder.zs(state)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, normalized_action)
           



        current_q1, current_q2 = self.critic(state, normalized_action, fixed_zsa, fixed_zs)
       
        td_error = torch.cat([(current_q1 - target_q).abs(), (current_q2 - target_q).abs()], dim=1)
        critic_loss = lap_huber(td_error, min_priority=self.min_priority) 

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        priority = td_error.max(1)[0].clamp(min=self.min_priority).pow(self.alpha)
        self.replay_buffer.update_priority(priority)

        if self.total_it % self.policy_freq == 0:
            actor_action = self.actor(state, fixed_zs)
            actor_zsa = self.fixed_encoder.zsa(fixed_zs, actor_action)
            actor_q1, actor_q2 = self.critic(state, actor_action, actor_zsa, fixed_zs)
            actor_loss = -torch.min(actor_q1, actor_q2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        if self.total_it % self.target_update_rate == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())
            self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            self.replay_buffer.reset_max_priority()
            self.max_target = self.max_q_uptodate
            self.min_target = self.min_q_uptodate

       

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.encoder.state_dict(), filename + "_encoder")
        torch.save(self.encoder_optimizer.state_dict(), filename + "_encoder_optimizer")
        torch.save(self.fixed_encoder.state_dict(), filename + "_fixed_encoder")
        torch.save(self.fixed_encoder_target.state_dict(), filename + "_fixed_encoder_target")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)

        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)

        self.encoder.load_state_dict(torch.load(filename + "_encoder", map_location=self.device))
        self.encoder_optimizer.load_state_dict(torch.load(filename + "_encoder_optimizer", map_location=self.device))
        self.fixed_encoder.load_state_dict(torch.load(filename + "_fixed_encoder", map_location=self.device))
        self.fixed_encoder_target.load_state_dict(torch.load(filename + "_fixed_encoder_target", map_location=self.device))
