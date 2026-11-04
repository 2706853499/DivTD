import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def quantile_huber_loss_f(quantiles, samples):
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]
    abs_pairwise_delta = pairwise_delta.abs()
    huber_loss = torch.where(
        abs_pairwise_delta > 1.0,
        abs_pairwise_delta - 0.5,
        0.5 * pairwise_delta.pow(2),
    )

    n_quantiles = quantiles.shape[2]
    tau = (
        torch.arange(n_quantiles, device=quantiles.device, dtype=torch.float32) / n_quantiles
        + 0.5 / n_quantiles
    )
    loss = (
        torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss
    ).mean()
    return loss


class Mlp(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Mlp, self).__init__()
        self.fcs = nn.ModuleList()
        in_size = input_size
        for next_size in hidden_sizes:
            self.fcs.append(nn.Linear(in_size, next_size))
            in_size = next_size
        self.last_fc = nn.Linear(in_size, output_size)

    def forward(self, x):
        h = x
        for fc in self.fcs:
            h = F.relu(fc(h))
        return self.last_fc(h)


class TanhNormal(Distribution):
    arg_constraints = {}

    def __init__(self, normal_mean, normal_std):
        super(TanhNormal, self).__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.standard_normal = Normal(
            torch.zeros_like(self.normal_mean),
            torch.ones_like(self.normal_std),
        )
        self.normal = Normal(normal_mean, normal_std)

    def log_prob(self, pre_tanh):
        log_det = 2 * np.log(2) + F.logsigmoid(2 * pre_tanh) + F.logsigmoid(-2 * pre_tanh)
        return self.normal.log_prob(pre_tanh) - log_det

    def rsample(self):
        pre_tanh = self.normal_mean + self.normal_std * self.standard_normal.sample()
        return torch.tanh(pre_tanh), pre_tanh


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_quantiles, n_nets):
        super(Critic, self).__init__()
        self.n_quantiles = int(n_quantiles)
        self.n_nets = int(n_nets)
        self.nets = nn.ModuleList(
            [
                Mlp(state_dim + action_dim, [512, 512, 512], self.n_quantiles)
                for _ in range(self.n_nets)
            ]
        )

    def forward(self, state, action):
        sa = torch.cat((state, action), dim=1)
        quantiles = torch.stack([net(sa) for net in self.nets], dim=1)
        return quantiles


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.net = Mlp(state_dim, [256, 256], 2 * action_dim)

    def forward(self, obs, deterministic=False):
        mean, log_std = self.net(obs).split([self.action_dim, self.action_dim], dim=1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            std = log_std.exp()
            tanh_normal = TanhNormal(mean, std)
            action, pre_tanh = tanh_normal.rsample()
            log_prob = tanh_normal.log_prob(pre_tanh).sum(dim=1, keepdim=True)

        return action * self.max_action, log_prob

    def select_action(self, obs, device, deterministic=False):
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
        action, _ = self.forward(obs, deterministic=deterministic)
        return action[0].cpu().detach().numpy()


class TQC(object):
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
        n_quantiles=25,
        n_nets=5,
        top_quantiles_to_drop_per_net=2,
        **kwargs
    ):
        self.device = device
        self.discount = discount
        self.tau = lamda
        self.target_entropy = -float(action_dim)
        self.total_it = 0

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim, n_quantiles, n_nets).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.log_alpha = torch.zeros((1,), requires_grad=True, device=self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        self.n_quantiles = int(n_quantiles)
        self.n_nets = int(n_nets)
        self.top_quantiles_to_drop = int(top_quantiles_to_drop_per_net) * int(n_nets)
        self.quantiles_total = self.n_quantiles * self.n_nets

    def select_action(self, state, deterministic=True):
        return self.actor.select_action(state, self.device, deterministic=deterministic)
    
    def q_bin_frequency(self, q: torch.Tensor, num_bins: int = 25):
        """
        Args:
            q: Tensor of shape (batch_size, 5, 25), values in [0,1]
            num_bins: number of bins (default 25)

        Returns:
            freq: Tensor of shape (batch_size, 5, 25)
                each entry is percentage of values falling into each bin
        """

        assert q.dim() == 3, "Q must be (B, 5, 25)"
        B, A, N = q.shape
        assert N == num_bins, "Last dim must match num_bins"

        # bin edges: [0, 1/25, 2/25, ..., 1]
        bin_edges = torch.linspace(0.0, 1.0, steps=num_bins + 1, device=q.device)

        # flatten for histogram computation
        q_flat = q.reshape(-1)

        # digitize into bins: returns index in [0, num_bins-1]
        bin_indices = torch.bucketize(q_flat, bin_edges, right=False) - 1
        bin_indices = bin_indices.clamp(0, num_bins - 1)

        # count occurrences
        hist = torch.bincount(bin_indices, minlength=num_bins).float()

        # normalize to percentage
        freq = hist / hist.sum().clamp(min=1.0)

        # reshape back if needed (same distribution for all batch entries)
        freq = freq.view(1,  num_bins).expand(B* A, num_bins)

        return freq.mean(dim=0).detach().cpu().numpy()
    def train(self, replay_buffer, batch_size=256, env=None, sz=None):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        alpha = torch.exp(self.log_alpha)

        with torch.no_grad():
            new_next_action, next_log_pi = self.actor(next_state, deterministic=False)
            next_z = self.critic_target(next_state, new_next_action)
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1), dim=1)
            sorted_z_part = sorted_z[:, : self.quantiles_total - self.top_quantiles_to_drop]
            target = reward + not_done * self.discount * (sorted_z_part - alpha * next_log_pi)

        cur_z = self.critic(state, action)
        critic_loss = quantile_huber_loss_f(cur_z, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

       
        new_action, log_pi = self.actor(state, deterministic=False)
        q_new_action = self.critic(state, new_action).mean(dim=2).mean(dim=1, keepdim=True)
        actor_loss = (alpha.detach() * log_pi - q_new_action).mean()
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

       

        self.total_it += 1
        if self.total_it %10==5:
            return self.q_bin_frequency(cur_z)
        else:
            return None

    def save(self, filename):
        filename = str(filename)
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_target.state_dict(), filename + "_critic_target")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.log_alpha.detach().cpu(), filename + "_log_alpha")
        torch.save(self.alpha_optimizer.state_dict(), filename + "_alpha_optimizer")

    def load(self, filename):
        filename = str(filename)
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=self.device))
        try:
            self.critic_target.load_state_dict(
                torch.load(filename + "_critic_target", map_location=self.device)
            )
        except FileNotFoundError:
            self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer", map_location=self.device)
        )
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer", map_location=self.device)
        )
        log_alpha = torch.load(filename + "_log_alpha", map_location=self.device)
        self.log_alpha.data.copy_(log_alpha.to(self.device))
        self.alpha_optimizer.load_state_dict(
            torch.load(filename + "_alpha_optimizer", map_location=self.device)
        )



