#!/usr/bin/env python3
#
# DDPG (Lillicrap et al., 2016) for TurtleBot3 goal-reaching.
#
# Deterministic actor mu(s) -> action, critic Q(s,a). Off-policy: trained from a
# replay buffer with target networks and Polyak averaging. Exploration is added
# by Ornstein-Uhlenbeck noise on the actor output during training only.

import numpy as np
import os

import torch
import torch.nn as nn

_USE_LAYERNORM = os.environ.get('DRL_LAYERNORM', '0') == '1'   # critic LayerNorm: tames Q-overestimation/collapse (env: DRL_LAYERNORM). Changes critic weights -> from scratch.

from ...common.ounoise import OUNoise
from ..base import OffPolicyAgent, Network

LINEAR = 0
ANGULAR = 1


class Actor(Network):
    # 44 -> 512 -> 512 -> 2, tanh output so actions land in [-1, 1] exactly as the
    # environment expects (it scales a0 by 0.22 m/s and a1 by 2.0 rad/s).
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Actor, self).__init__(name)
        self.fa1 = nn.Linear(state_size, hidden_size)
        self.fa2 = nn.Linear(hidden_size, hidden_size)
        self.fa3 = nn.Linear(hidden_size, action_size)
        self.apply(Network.init_weights)

    def forward(self, states):
        x = torch.relu(self.fa1(states))
        x = torch.relu(self.fa2(x))
        return torch.tanh(self.fa3(x))


class Critic(Network):
    # The 40-dim lidar dominates the state, so the 2-dim action gets its own
    # projection (state->256, action->256) before they are concatenated; this
    # keeps the action's influence on Q from being washed out early.
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Critic, self).__init__(name)
        self.l1 = nn.Linear(state_size, int(hidden_size / 2))
        self.l2 = nn.Linear(action_size, int(hidden_size / 2))
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, 1)
        self.use_ln = _USE_LAYERNORM
        if self.use_ln:                       # pre-activation LayerNorm (SpinningUp/Fujimoto recipe)
            self.ln1 = nn.LayerNorm(int(hidden_size / 2))
            self.ln2 = nn.LayerNorm(int(hidden_size / 2))
            self.ln3 = nn.LayerNorm(hidden_size)
        self.apply(Network.init_weights)      # only touches nn.Linear; LN affine params stay 1/0

    def forward(self, states, actions):
        xs = self.l1(states); xs = torch.relu(self.ln1(xs) if self.use_ln else xs)
        xa = self.l2(actions); xa = torch.relu(self.ln2(xa) if self.use_ln else xa)
        x = torch.cat((xs, xa), dim=1)
        x = self.l3(x); x = torch.relu(self.ln3(x) if self.use_ln else x)
        return self.l4(x)


class DDPG(OffPolicyAgent):
    def __init__(self, device, sim_speed):
        super().__init__(device, sim_speed)

        # Constant-sigma OU noise: correlated exploration gives smoother, more
        # committed velocity commands than white noise on a differential drive.
        from ...common.settings import NOISE_SIGMA_MAX, NOISE_SIGMA_MIN, NOISE_DECAY_PERIOD
        self.noise = OUNoise(action_space=self.action_size, max_sigma=NOISE_SIGMA_MAX,
                             min_sigma=NOISE_SIGMA_MIN, decay_period=NOISE_DECAY_PERIOD)

        self.actor = self.create_network(Actor, 'actor')
        self.actor_target = self.create_network(Actor, 'target_actor')
        self.actor_optimizer = self.create_optimizer(self.actor)

        self.critic = self.create_network(Critic, 'critic')
        self.critic_target = self.create_network(Critic, 'target_critic')
        self.critic_optimizer = self.create_optimizer(self.critic)

        # Targets start identical to the online networks.
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

    def get_action(self, state, is_training, step, visualize=False):
        state = torch.from_numpy(np.asarray(state, np.float32)).to(self.device)
        action = self.actor(state)
        if is_training:
            noise = torch.from_numpy(self.noise.get_noise(step)).to(self.device)
            action = torch.clamp(action + noise, -1.0, 1.0)
        return action.detach().cpu().numpy().tolist()

    def get_action_random(self):
        # Independent uniform sample PER action dimension. (The reference code
        # replicated a single scalar across both dims, so during the 25k-step
        # observe phase linear and angular were always identical -- crippling the
        # diversity of the seed data. Sampling per-dim fixes that.)
        return np.random.uniform(-1.0, 1.0, self.action_size).astype(np.float32).tolist()

    def train(self, state, action, reward, state_next, done, weights=None):
        # --- Critic: regress Q(s,a) onto the one-step TD target ---
        with torch.no_grad():
            action_next = self.actor_target(state_next)
            Q_next = self.critic_target(state_next, action_next)
            # `reward` is the n-step discounted sum (n=1 -> single reward), `state_next` is
            # s_{t+n}, so the bootstrap discount is gamma**n (n=1 -> gamma). getattr fallback
            # keeps checkpoints pickled before n-step was added working.
            nstep_discount = getattr(self, 'nstep_discount', self.discount_factor)
            Q_target = reward + (1 - done) * nstep_discount * Q_next
        Q = self.critic(state, action)

        td_error = Q - Q_target                          # per-sample TD error (for PER priorities)
        if weights is not None:                          # PER: importance-sampling-weighted MSE
            loss_critic = (weights * td_error.pow(2)).mean()
        else:
            loss_critic = self.loss_function(Q, Q_target)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optimizer.step()

        # --- Actor: deterministic policy gradient, ascend Q(s, mu(s)) ---
        loss_actor = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
        self.actor_optimizer.step()

        # --- Track target networks ---
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)

        out = [loss_critic.mean().detach().cpu(), loss_actor.mean().detach().cpu()]
        if weights is not None:
            out.append(td_error.detach().squeeze(-1).cpu().numpy())   # for update_priorities
        return out
