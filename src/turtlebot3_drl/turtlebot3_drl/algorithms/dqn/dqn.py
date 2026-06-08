import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from ...common.settings import DQN_ACTION_SIZE, TARGET_UPDATE_FREQUENCY

from ..base import OffPolicyAgent, Network

LINEAR = 0
ANGULAR = 1

# Discrete action set: each entry is [linear, angular] in [-1, 1] (env un-normalises to m/s, rad/s).
# get_action returns the INDEX into this list; to_env_action maps the index back to the pair.
POSSIBLE_ACTIONS = [[0.3, -1.0], [0.3, -0.5], [1.0, 0.0], [0.3, 0.5], [0.3, 1.0]]

# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance/blob/main/turtlebot_ddpg/scripts/original_ddpg/ddpg_network_turtlebot3_original_ddpg.py

class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Actor, self).__init__(name)

        # --- define layers here ---
        self.fa1 = nn.Linear(state_size, hidden_size)
        self.fa2 = nn.Linear(hidden_size, hidden_size)
        self.fa3 = nn.Linear(hidden_size, action_size)
        # --- define layers until here ---

        self.apply(super().init_weights)

    def forward(self, states, visualize=False):
        # --- define forward pass here ---
        x1 = torch.relu(self.fa1(states))
        x2 = torch.relu(self.fa2(x1))
        action = self.fa3(x2)
        # --- define forward pass until here ---

        # -- define layers to visualize here (optional) ---
        if visualize and self.visual:
            action = torch.from_numpy(np.asarray(POSSIBLE_ACTIONS[action.argmax().tolist()], np.float32))
            self.visual.update_layers(states, action, [x1, x2], [self.fa1.bias, self.fa2.bias])
        # -- define layers to visualize until here ---
        return action

class DQN(OffPolicyAgent):
    def __init__(self, device, sim_speed):
        super().__init__(device, sim_speed)

        # Override the continuous action_size BEFORE building the network so the Q-head has
        # one output per discrete action.
        self.action_size = DQN_ACTION_SIZE
        self.possible_actions = POSSIBLE_ACTIONS
        self.target_update_frequency = TARGET_UPDATE_FREQUENCY

        self.actor = self.create_network(Actor, 'actor')
        self.actor_target = self.create_network(Actor, 'target_actor')
        self.actor_optimizer = self.create_optimizer(self.actor)

        self.hard_update(self.actor_target, self.actor)

    def to_env_action(self, action):
        # Map the discrete action index the policy chose to the [linear, angular] the env drives.
        return self.possible_actions[int(action)]

    def get_action(self, state, is_training, step=0, visualize=False):
        if is_training and np.random.random() < self.epsilon:
            return self.get_action_random()
        state = torch.from_numpy(np.asarray(state, np.float32)).to(self.device)
        Q_values = self.actor(state, visualize).detach().cpu()
        return int(Q_values.argmax().item())

    def get_action_random(self):
        return int(np.random.randint(0, self.action_size))

    def train(self, state, action, reward, state_next, done, weights=None):
        # action arrives as float (batch,) from the buffer (the stored discrete index) -> long index.
        action = action.long().unsqueeze(1)
        Q_next = self.actor_target(state_next).amax(1, keepdim=True)
        Q_target = reward + (self.discount_factor * Q_next * (1 - done))
        # Select the Q-value of the action that was actually taken in each sampled state.
        Q = self.actor(state).gather(1, action)
        td_error = Q - Q_target
        # PER (optional): importance-sampling-weighted loss; plain MSE otherwise.
        loss = (weights * td_error.pow(2)).mean() if weights is not None else td_error.pow(2).mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0, norm_type=1)
        self.actor_optimizer.step()

        # Hard target sync every N train iterations.
        if self.iteration % self.target_update_frequency == 0:
            self.hard_update(self.actor_target, self.actor)

        # (loss_critic, loss_actor, per_sample_|TD error|) — the 3rd is consumed by PER only.
        return 0, loss.detach().cpu(), td_error.abs().detach().cpu().numpy().flatten()
