#!/usr/bin/env python3
#
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Base class for the off-policy actor-critic agent (DDPG).
# It owns the hyperparameters, the numpy->torch batch plumbing, network/optimizer
# factories, and the target-network update rules. Algorithm-specific logic
# (network shapes, action selection, the loss) lives in the subclass `train()`.

from abc import ABC, abstractmethod
import torch
import torch.nn.functional as torchf

from turtlebot3_drl.drl_environment.reward import REWARD_FUNCTION
from ..common.settings import ENABLE_BACKWARD, ENABLE_STACKING, ACTION_SIZE, HIDDEN_SIZE, BATCH_SIZE, BUFFER_SIZE, \
    DISCOUNT_FACTOR, LEARNING_RATE, TAU, STEP_TIME, EPSILON_DECAY, EPSILON_MINIMUM, STACK_DEPTH, FRAME_SKIP, N_STEP
from ..drl_environment.drl_environment import NUM_SCAN_SAMPLES


class OffPolicyAgent(ABC):
    def __init__(self, device, simulation_speed):
        self.device = device
        self.simulation_speed = simulation_speed

        # Network shape, fixed by the environment:
        #   state  = NUM_SCAN_SAMPLES lidar bins (40) + goal_distance + goal_angle
        #            + previous_linear + previous_angular  ->  44 dims
        #   action = [linear, angular], each in [-1, 1] (env un-normalises to m/s, rad/s)
        self.state_size = NUM_SCAN_SAMPLES + 4
        self.action_size = ACTION_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.input_size = self.state_size

        # Hyperparameters
        self.batch_size = BATCH_SIZE
        self.buffer_size = BUFFER_SIZE
        self.discount_factor = DISCOUNT_FACTOR
        # N-step: bootstrap discount is gamma**n (computed once; n=1 -> gamma == 1-step).
        self.n_step = N_STEP
        self.nstep_discount = self.discount_factor ** self.n_step
        self.learning_rate = LEARNING_RATE
        self.tau = TAU

        # Control cadence + misc
        self.step_time = STEP_TIME
        # Huber loss for the critic: robust to the large, spiky TD targets that the
        # +2500 / -2000 terminal rewards produce (with REWARD_SCALE=1.0).
        self.loss_function = torchf.smooth_l1_loss
        self.epsilon = 1.0
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_minimum = EPSILON_MINIMUM
        self.reward_function = REWARD_FUNCTION
        self.backward_enabled = ENABLE_BACKWARD
        self.stacking_enabled = ENABLE_STACKING
        self.stack_depth = STACK_DEPTH
        self.frame_skip = FRAME_SKIP
        if ENABLE_STACKING:
            self.input_size *= self.stack_depth

        self.networks = []
        self.iteration = 0

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def get_action(self):
        pass

    @abstractmethod
    def get_action_random(self):
        pass

    def _train(self, replaybuffer):
        # Pull a minibatch and move it to the GPU once. With PER the buffer also returns
        # the sampled indices + importance-sampling weights (used to de-bias the loss),
        # and we write the fresh |TD error| back as the new priorities.
        per = hasattr(replaybuffer, 'update_priorities')
        if per:
            sample_s, sample_a, sample_r, sample_ns, sample_d, idxs, is_w = replaybuffer.sample(self.batch_size)
            weights = torch.from_numpy(is_w).to(self.device).view(-1, 1)
        else:
            sample_s, sample_a, sample_r, sample_ns, sample_d = replaybuffer.sample(self.batch_size)
            weights = None
        sample_s = torch.from_numpy(sample_s).to(self.device)
        sample_a = torch.from_numpy(sample_a).to(self.device)
        sample_r = torch.from_numpy(sample_r).to(self.device)
        sample_ns = torch.from_numpy(sample_ns).to(self.device)
        sample_d = torch.from_numpy(sample_d).to(self.device)
        result = self.train(sample_s, sample_a, sample_r, sample_ns, sample_d, weights)
        if per:
            replaybuffer.update_priorities(idxs, result[2])   # result[2] = per-sample |TD error|
        self.iteration += 1
        if self.epsilon and self.epsilon > self.epsilon_minimum:
            self.epsilon *= self.epsilon_decay
        return result[:2]

    def create_network(self, type, name):
        network = type(name, self.input_size, self.action_size, self.hidden_size).to(self.device)
        self.networks.append(network)
        return network

    def create_optimizer(self, network):
        return torch.optim.AdamW(network.parameters(), self.learning_rate)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        # Polyak averaging: target slowly tracks the online net (tau from settings, 0.003).
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_model_configuration(self):
        configuration = ""
        for attribute, value in self.__dict__.items():
            if attribute not in ['actor', 'actor_target', 'critic', 'critic_target']:
                configuration += f"{attribute} = {value}\n"
        return configuration

    def get_model_parameters(self):
        parameters = [self.batch_size, self.buffer_size, self.state_size, self.action_size, self.hidden_size,
                      self.discount_factor, self.learning_rate, self.tau, self.step_time, REWARD_FUNCTION,
                      ENABLE_BACKWARD, ENABLE_STACKING, self.stack_depth, self.frame_skip]
        return ', '.join(map(str, parameters))


class Network(torch.nn.Module, ABC):
    def __init__(self, name):
        super(Network, self).__init__()
        self.name = name
        self.iteration = 0

    @abstractmethod
    def forward(self):
        pass

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
