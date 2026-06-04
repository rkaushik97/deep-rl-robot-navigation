#!/usr/bin/env python3
#
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Authors: Ryan Shim, Gilbert, Tomas
#
# The learner node. Owns the networks + replay buffer and runs the act-learn loop:
#   wait for goal -> reset episode -> {act, step, store, train} until terminal ->
#   log / checkpoint / validate.
# Acting talks to the environment node over the `step_comm` service; the gz world is
# paused around every gradient update so the robot does not drift on its last command
# while the GPU works (gz is wall-clock paced and uncapped).

import copy
import os
import sys
import time
import json
import torch

from ..common.settings import OBSERVE_STEPS, MODEL_STORE_INTERVAL, GRAPH_DRAW_INTERVAL, VAL_EPS_PER_CHECKPOINT, MAX_TRAINING_EPISODES

from ..common.storagemanager import StorageManager
from ..common.graph import Graph
from ..common.logger import Logger
from ..common import utilities as util
from ..common.replaybuffer import ReplayBuffer

from .ddpg import DDPG

from turtlebot3_msgs.srv import DrlStep, Goal
from ros_gz_interfaces.srv import ControlWorld
from std_msgs.msg import String

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

SUCCESS = 1  # outcome code (see utilities.translate_outcome)


class DrlAgent(Node):
    def __init__(self, training, algorithm, load_session="", load_episode=0, real_robot=0):
        super().__init__(algorithm + '_agent')
        self.algorithm = algorithm
        self.training = int(training)
        self.load_session = load_session
        self.episode = int(load_episode)
        self.real_robot = real_robot

        if (not self.training and not self.load_session):
            quit("Testing requested but no model to load specified (see README for the command format)")

        self.device = util.check_gpu()
        self.sim_speed = util.get_simulation_speed(util.stage) if not self.real_robot else 1
        print(f"{'training' if self.training else 'testing'} on stage: {util.stage}")
        self.total_steps = 0
        self.observe_steps = OBSERVE_STEPS

        if self.algorithm == 'ddpg':
            self.model = DDPG(self.device, self.sim_speed)
        else:
            quit(f"invalid algorithm specified ({self.algorithm}); this build supports: ddpg")

        self.replay_buffer = ReplayBuffer(self.model.buffer_size)
        self.graph = Graph()

        # Eval-based checkpoint selection: keep the best deterministic-policy score
        # seen so far, decoupled from the noisy stochastic training reward.
        self.best_val_score = -1.0
        self.best_val_episode = 0

        # ----- model loading / new session -----
        self.sm = StorageManager(self.algorithm, self.load_session, self.episode, self.device, util.stage)
        if self.load_session:
            del self.model
            self.model = self.sm.load_model()
            self.model.device = self.device
            self.sm.load_weights(self.model.networks)
            if self.training:
                self.replay_buffer.buffer = self.sm.load_replay_buffer(
                    self.model.buffer_size, os.path.join(self.load_session, 'stage' + str(self.sm.stage) + '_latest_buffer.pkl'))
            self.graph.session_dir = self.sm.session_dir
            self.total_steps = self.graph.set_graphdata(self.sm.load_graphdata(), self.episode)
            print(f"global steps: {self.total_steps}")
            print(f"loaded model {self.load_session} (eps {self.episode}): {self.model.get_model_parameters()}")
        else:
            self.sm.new_session_dir(util.stage)
            self.sm.store_model(self.model)

        self.graph.session_dir = self.sm.session_dir
        self.logger = Logger(self.training, self.sm.machine_dir, self.sm.session_dir, self.sm.session,
                             self.model.get_model_parameters(), self.model.get_model_configuration(),
                             str(util.stage), self.algorithm, self.episode)

        # ----- ROS clients / optional debug -----
        self.step_comm_client = self.create_client(DrlStep, 'step_comm')
        self.goal_comm_client = self.create_client(Goal, 'goal_comm')

        self.debug_enabled = os.environ.get('DEBUG_DRL', '0') == '1'
        self.debug_pub = (self.create_publisher(String, '/drl_debug/train', QoSProfile(depth=10))
                          if self.debug_enabled else None)
        if self.debug_enabled:
            print("[DEBUG_DRL] agent publishing per-train-step debug JSON on /drl_debug/train")

        if not self.real_robot:
            world_name = f'drl_stage{util.stage}'
            self.gazebo_control = self.create_client(ControlWorld, f'/world/{world_name}/control')

        self.process()

    # ===================================================================== #
    #                            Main act-learn loop                        #
    # ===================================================================== #
    def process(self):
        util.pause_simulation(self, self.real_robot)
        while True:
            util.wait_new_goal(self)
            episode_done = False
            step, reward_sum, loss_critic, loss_actor = 0, 0, 0, 0
            action_past = [0.0, 0.0]
            state = util.init_episode(self)

            util.unpause_simulation(self, self.real_robot)
            time.sleep(0.5)
            episode_start = time.perf_counter()

            while not episode_done:
                # Random actions during the observe phase to seed the buffer with
                # diverse data; greedy actor + OU noise afterwards.
                if self.training and self.total_steps < self.observe_steps:
                    action = self.model.get_action_random()
                else:
                    action = self.model.get_action(state, self.training, step)

                next_state, reward, episode_done, outcome, distance_traveled, _ = util.step(self, action, action_past)
                action_past = copy.deepcopy(action)
                reward_sum += reward

                if self.training:
                    self.replay_buffer.add_sample(state, action, [reward], next_state, [episode_done])
                    if self.replay_buffer.get_length() >= self.model.batch_size:
                        # Freeze the world during the gradient update (see module docstring).
                        util.pause_simulation(self, self.real_robot)
                        loss_c, loss_a = self.model._train(self.replay_buffer)
                        util.unpause_simulation(self, self.real_robot)
                        loss_critic += loss_c
                        loss_actor += loss_a
                        if self.debug_pub is not None:
                            self.debug_pub.publish(String(data=json.dumps({
                                'episode': int(self.episode),
                                'total_steps': int(self.total_steps + step),
                                'buffer_len': int(self.replay_buffer.get_length()),
                                'loss_critic': float(loss_c),
                                'loss_actor': float(loss_a),
                                'iteration': int(self.model.iteration),
                                'epsilon': float(self.model.epsilon),
                                'action_last': [float(a) for a in action],
                                'reward_last': float(reward),
                            })))

                state = copy.deepcopy(next_state)
                step += 1
                time.sleep(self.model.step_time)

            # Episode finished
            util.pause_simulation(self, self.real_robot)
            self.total_steps += step
            duration = time.perf_counter() - episode_start
            self.finish_episode(step, duration, outcome, distance_traveled, reward_sum, loss_critic, loss_actor)

            # Experiment episode budget (env: DRL_MAX_EPISODES). Clean stop for "train N eps then analyze".
            if MAX_TRAINING_EPISODES and self.training and self.episode >= MAX_TRAINING_EPISODES:
                print(f"[done] reached MAX_TRAINING_EPISODES={MAX_TRAINING_EPISODES} — saving + stopping.", flush=True)
                self.sm.save_session(self.episode, self.model.networks, self.graph.graphdata, self.replay_buffer.buffer)
                self.graph.draw_plots(self.episode)
                sys.exit(0)

    def finish_episode(self, step, eps_duration, outcome, dist_traveled, reward_sum, loss_critic, loss_actor):
        if self.total_steps < self.observe_steps:
            print(f"Observe phase: {self.total_steps}/{self.observe_steps} steps")
            return

        self.episode += 1
        print(f"Epi: {self.episode:<5}R: {reward_sum:<8.0f}outcome: {util.translate_outcome(outcome):<13}"
              f"steps: {step:<6}steps_total: {self.total_steps:<7}time: {eps_duration:<6.2f}")

        if not self.training:
            self.logger.update_test_results(step, outcome, dist_traveled, eps_duration, 0)
            return

        self.graph.update_data(step, self.total_steps, outcome, reward_sum, loss_critic, loss_actor)
        self.logger.file_log.write(
            f"{self.episode}, {reward_sum}, {outcome}, {eps_duration}, {step}, {self.total_steps}, "
            f"{self.replay_buffer.get_length()}, {loss_critic / step}, {loss_actor / step}\n")

        if (self.episode % MODEL_STORE_INTERVAL == 0) or (self.episode == 1):
            self.sm.save_session(self.episode, self.model.networks, self.graph.graphdata, self.replay_buffer.buffer)
            self.logger.update_comparison_file(self.episode, self.graph.get_success_count(), self.graph.get_reward_average())
            # ---- validation every MODEL_STORE_INTERVAL episodes ----
            if self.episode > 1 and not self.real_robot and VAL_EPS_PER_CHECKPOINT > 0:
                val_score = self._run_validation_eval(VAL_EPS_PER_CHECKPOINT)
                print(f"[val] ep {self.episode}: {val_score*100:.1f}% over {VAL_EPS_PER_CHECKPOINT} det. eps  "
                      f"|  best: {self.best_val_score*100:.1f}% @ ep {self.best_val_episode}")
                if val_score > self.best_val_score:
                    self.best_val_score = val_score
                    self.best_val_episode = self.episode
                    self._save_best_checkpoint(val_score)
                    print(f"[val] NEW BEST CHECKPOINT — ep {self.episode}, success {val_score*100:.1f}%")
                # Log the deterministic eval curve next to the other run artifacts
                # (same dir as _figure.png / _train_*.txt).
                eval_path = os.path.join(self.sm.session_dir, f"_eval_stage{self.sm.stage}.tsv")
                if not os.path.exists(eval_path):
                    with open(eval_path, 'w') as f:
                        f.write("episode\ttotal_steps\tval_eps\tval_success\tbest_success\tbest_episode\n")
                with open(eval_path, 'a') as f:
                    f.write(f"{self.episode}\t{self.total_steps}\t{VAL_EPS_PER_CHECKPOINT}\t"
                            f"{val_score:.4f}\t{self.best_val_score:.4f}\t{self.best_val_episode}\n")
        if (self.episode % GRAPH_DRAW_INTERVAL == 0) or (self.episode == 1):
            self.graph.draw_plots(self.episode)

    def _run_validation_eval(self, num_eps):
        """num_eps deterministic-policy episodes (no exploration noise, no buffer
        adds, no gradient updates). Returns success rate in [0, 1]."""
        successes = 0
        for _ in range(num_eps):
            util.wait_new_goal(self)
            episode_done = False
            step = 0
            action_past = [0.0, 0.0]
            state = util.init_episode(self)
            util.unpause_simulation(self, self.real_robot)
            time.sleep(0.5)
            outcome = 0
            while not episode_done:
                action = self.model.get_action(state, False, step)  # is_training=False -> greedy, no noise
                next_state, _, episode_done, outcome, _, _ = util.step(self, action, action_past)
                action_past = copy.deepcopy(action)
                state = copy.deepcopy(next_state)
                step += 1
                time.sleep(self.model.step_time)
            util.pause_simulation(self, self.real_robot)
            if outcome == SUCCESS:
                successes += 1
        return successes / num_eps if num_eps > 0 else 0.0

    def _save_best_checkpoint(self, score):
        for network in self.model.networks:
            filepath = os.path.join(self.sm.session_dir, f"{network.name}_stage{self.sm.stage}_best.pt")
            torch.save(network.state_dict(), filepath)
        with open(os.path.join(self.sm.session_dir, '_best_metadata.txt'), 'w') as f:
            f.write(f"episode={self.episode}\n")
            f.write(f"success_rate={score:.4f}\n")
            f.write(f"timestamp={time.strftime('%Y%m%d-%H%M%S')}\n")


def main(args=sys.argv[1:]):
    rclpy.init(args=args)
    drl_agent = DrlAgent(*args)
    rclpy.spin(drl_agent)
    drl_agent.destroy()
    rclpy.shutdown()


def main_train(args=sys.argv[1:]):
    main(['1'] + args)


def main_test(args=sys.argv[1:]):
    main(['0'] + args)


if __name__ == '__main__':
    main()
