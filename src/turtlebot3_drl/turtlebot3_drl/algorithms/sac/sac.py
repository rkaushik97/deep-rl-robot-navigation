import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...common.settings import (
    SAC_ALPHA_LR,
    SAC_INIT_LOG_ALPHA,
    SAC_LOG_STD_MAX,
    SAC_LOG_STD_MIN,
    SAC_TARGET_ENTROPY_SCALE,
)

from ..base import Network, OffPolicyAgent

LINEAR = 0
ANGULAR = 1

# Soft Actor-Critic (Haarnoja et al. 2018 v2 — auto-tuned temperature α).
# Structure mirrors td3.py for a near drop-in swap:
#   * Twin Q-critics with a single target network — no actor_target since
#     SAC's stochastic policy doesn't need one.
#   * Tanh-squashed Gaussian actor; log-prob uses the standard tanh
#     correction term log(1 - tanh(u)^2 + eps).
#   * α is a learned scalar (parameterised as log_alpha) tuned so policy
#     entropy tracks `target_entropy = -|A| * scale`.


class GaussianActor(Network):
    """Tanh-squashed Gaussian policy — outputs an action in [-1, 1]^A."""

    def __init__(self, name, state_size, action_size, hidden_size):
        super().__init__(name)
        self.fa1 = nn.Linear(state_size, hidden_size)
        self.fa2 = nn.Linear(hidden_size, hidden_size)
        self.fa_mu = nn.Linear(hidden_size, action_size)
        self.fa_log_std = nn.Linear(hidden_size, action_size)
        self.log_std_min = SAC_LOG_STD_MIN
        self.log_std_max = SAC_LOG_STD_MAX

        self.apply(super().init_weights)

    def forward(self, states, visualize=False):
        x1 = F.relu(self.fa1(states))
        x2 = F.relu(self.fa2(x1))
        mu = self.fa_mu(x2)
        log_std = self.fa_log_std(x2).clamp(self.log_std_min, self.log_std_max)

        if visualize and self.visual:
            self.visual.update_layers(
                states, torch.tanh(mu), [x1, x2], [self.fa1.bias, self.fa2.bias]
            )
        return mu, log_std

    def sample(self, states):
        mu, log_std = self.forward(states)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        u = normal.rsample()                                   # reparam trick
        action = torch.tanh(u)
        # tanh correction (SAC paper appendix C).
        log_pi = normal.log_prob(u) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_pi = log_pi.sum(-1, keepdim=True)
        return action, log_pi


class Critic(Network):
    """Twin Q(s, a) — identical structure to td3.py Critic for parity."""

    def __init__(self, name, state_size, action_size, hidden_size):
        super().__init__(name)
        half = int(hidden_size / 2)

        # Q1
        self.l1 = nn.Linear(state_size, half)
        self.l2 = nn.Linear(action_size, half)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, 1)

        # Q2
        self.l5 = nn.Linear(state_size, half)
        self.l6 = nn.Linear(action_size, half)
        self.l7 = nn.Linear(hidden_size, hidden_size)
        self.l8 = nn.Linear(hidden_size, 1)

        self.apply(super().init_weights)

    def forward(self, states, actions):
        xs = F.relu(self.l1(states))
        xa = F.relu(self.l2(actions))
        x = torch.cat((xs, xa), dim=1)
        x = F.relu(self.l3(x))
        q1 = self.l4(x)

        xs = F.relu(self.l5(states))
        xa = F.relu(self.l6(actions))
        x = torch.cat((xs, xa), dim=1)
        x = F.relu(self.l7(x))
        q2 = self.l8(x)

        return q1, q2


class SAC(OffPolicyAgent):
    def __init__(self, device, sim_speed):
        super().__init__(device, sim_speed)

        # Auto-α: keep mean policy entropy near -|A| * scale.
        self.target_entropy = -float(self.action_size) * SAC_TARGET_ENTROPY_SCALE
        self.log_alpha = torch.tensor(
            float(SAC_INIT_LOG_ALPHA),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=SAC_ALPHA_LR)
        self.last_actor_loss = 0.0
        self.last_alpha = float(self.log_alpha.exp().item())

        self.actor = self.create_network(GaussianActor, 'actor')
        self.actor_optimizer = self.create_optimizer(self.actor)

        self.critic = self.create_network(Critic, 'critic')
        self.critic_target = self.create_network(Critic, 'target_critic')
        self.critic_optimizer = self.create_optimizer(self.critic)

        self.hard_update(self.critic_target, self.critic)

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def get_action(self, state, is_training, step, visualize=False):
        state = torch.from_numpy(np.asarray(state, np.float32)).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            if is_training:
                action, _ = self.actor.sample(state)
            else:
                # At eval, take the deterministic tanh-squashed mean.
                mu, _ = self.actor(state, visualize)
                action = torch.tanh(mu)
        return action.squeeze(0).cpu().numpy().tolist()

    def get_action_random(self):
        return [float(np.clip(np.random.uniform(-1.0, 1.0), -1.0, 1.0))] * self.action_size

    def train(self, state, action, reward, state_next, done, weights=None):
        # weights accepted for API compat with off_policy_agent._train (PER); SAC replication
        # runs PER off so it's ignored. nstep_discount == discount_factor when N_STEP=1.
        # --- critic update -------------------------------------------------
        with torch.no_grad():
            a_next, logp_next = self.actor.sample(state_next)
            q1_next, q2_next = self.critic_target(state_next, a_next)
            q_next = torch.min(q1_next, q2_next) - self.alpha * logp_next
            nstep_discount = getattr(self, 'nstep_discount', self.discount_factor)
            q_target = reward + (1.0 - done) * nstep_discount * q_next

        q1, q2 = self.critic(state, action)
        loss_critic = self.loss_function(q1, q_target) + self.loss_function(q2, q_target)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optimizer.step()

        # --- actor update --------------------------------------------------
        a_new, logp_new = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, a_new)
        q_new = torch.min(q1_new, q2_new)
        loss_actor = (self.alpha * logp_new - q_new).mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
        self.actor_optimizer.step()

        # --- α update (auto-tune temperature) ------------------------------
        loss_alpha = -(self.log_alpha * (logp_new.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        loss_alpha.backward()
        self.alpha_optimizer.step()
        self.last_alpha = float(self.alpha.item())

        # --- polyak target update -----------------------------------------
        self.soft_update(self.critic_target, self.critic, self.tau)

        self.last_actor_loss = loss_actor.detach().cpu()
        return [loss_critic.detach().cpu(), self.last_actor_loss]
