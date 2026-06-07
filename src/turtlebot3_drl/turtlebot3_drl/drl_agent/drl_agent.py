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
from collections import deque
import time
import json
import torch

from ..common.settings import OBSERVE_STEPS, MODEL_STORE_INTERVAL, GRAPH_DRAW_INTERVAL, VAL_EPS_PER_CHECKPOINT, MAX_TRAINING_EPISODES, TEST_EPISODES, N_STEP

from ..common.storagemanager import StorageManager
from ..common.graph import Graph
from ..common.logger import Logger
from ..common import utilities as util
from ..common.replaybuffer import ReplayBuffer

from ..algorithms import get_agent_class
from ..training.metrics import TrainingMetrics
from ..training.display import episode_line
from ..training import plots
from ..drl_environment.reward import REWARD_COMPONENT_NAMES

from turtlebot3_msgs.srv import DrlStep, Goal
from ros_gz_interfaces.srv import ControlWorld
from std_msgs.msg import String

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

SUCCESS = 1  # outcome code (see utilities.translate_outcome)


class FrameStacker:
    """Per-episode temporal-observation assembler (frame-stacking for moving obstacles).
    Keeps the last (stack_depth-1)*frame_skip + 1 RAW frames and emits a flat vector
    concatenating frames at offsets [0, -frame_skip, ...] (newest first). reset() pads by
    repeating the first frame so step 0 already yields a full-width stacked vector. A single
    shared instance is fine: training and eval episodes are strictly sequential and each
    begins with reset()."""
    def __init__(self, stack_depth, frame_skip):
        self.stack_depth = stack_depth
        self.frame_skip = frame_skip
        self.maxlen = (stack_depth - 1) * frame_skip + 1
        self.buffer = deque(maxlen=self.maxlen)

    def reset(self, first_frame):
        f = list(first_frame)
        self.buffer.clear()
        for _ in range(self.maxlen):
            self.buffer.append(f)
        return self._assemble()

    def push(self, frame):
        self.buffer.append(list(frame))
        return self._assemble()

    def _assemble(self):
        b = list(self.buffer)
        idxs = [self.maxlen - 1 - k * self.frame_skip for k in range(self.stack_depth)]
        stacked = []
        for i in idxs:
            stacked.extend(b[i])
        return stacked


class DrlAgent(Node):
    def __init__(self, training, algorithm, load_session="", load_episode=0, real_robot=0):
        super().__init__(algorithm + '_agent')
        self.algorithm = algorithm
        self.training = int(training)
        self.load_session = load_session
        self.episode = int(load_episode)
        self.load_episode = int(load_episode)   # kept separate: self.episode is reused as a counter
        self.real_robot = real_robot

        if (not self.training and not self.load_session):
            quit("Testing requested but no model to load specified (see README for the command format)")

        self.device = util.check_gpu()
        self.sim_speed = util.get_simulation_speed(util.stage) if not self.real_robot else 1
        print(f"{'training' if self.training else 'testing'} on stage: {util.stage}")
        self.total_steps = 0
        self.observe_steps = OBSERVE_STEPS

        self.model = get_agent_class(self.algorithm)(self.device, self.sim_speed)

        if os.environ.get('DRL_PER', '0') == '1':
            from ..common.replaybuffer import PrioritizedReplayBuffer
            self.replay_buffer = PrioritizedReplayBuffer(self.model.buffer_size)
            print("[PER] prioritized experience replay ENABLED", flush=True)
        else:
            self.replay_buffer = ReplayBuffer(self.model.buffer_size)
        # Frame-stacking (temporal obs): assemble in the agent so the env stays unchanged and
        # the SAME stacked vector feeds both get_action and the replay buffer. Disabled by default.
        self.stacking = self.model.stacking_enabled
        self.frame_stacker = (FrameStacker(self.model.stack_depth, self.model.frame_skip)
                              if self.stacking else None)
        self.graph = Graph()

        # Eval-based checkpoint selection: keep the best deterministic-policy score
        # seen so far, decoupled from the noisy stochastic training reward.
        self.best_val_score = -1.0
        self.best_val_episode = 0

        # ----- model loading / new session -----
        self.sm = StorageManager(self.algorithm, self.load_session, self.episode, self.device, util.stage)
        if self.load_session and not self.training:
            # EVAL/TEST: weights-only load. We keep the freshly-built model and load just the
            # actor's .pt weights — no pickled agent (stage*_agent.pkl), optimizer state, or
            # buffer needed. Robust to checkpoints saved under any module layout / host.
            self.sm.load_weights([self.model.actor])
            self.graph.session_dir = self.sm.session_dir
            self.total_steps = self.observe_steps   # not in the random-observe phase during eval
            self.train_start_len = self.model.batch_size
            print(f"loaded actor weights {self.load_session} (ep {self.episode}) for evaluation")
        elif self.load_session:
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
            self.train_start_len = self.model.batch_size
        else:
            self.sm.new_session_dir(util.stage)
            self.train_start_len = self.model.batch_size
            warmstart = os.environ.get('DRL_WARMSTART', '')
            if warmstart:
                # WARM-START: load weights from a prior run's *_best.pt into the fresh
                # model, but keep a FRESH replay buffer + NEW session and SKIP the observe
                # phase (the loaded policy is already trained, so every new transition uses
                # the CURRENT reward function). Lets us add a dense reward component on top
                # of a working policy instead of re-bootstrapping from scratch.
                for net in self.model.networks:
                    fp = os.path.join(warmstart, f"{net.name}_stage{self.sm.stage}_best.pt")
                    sd = torch.load(fp, map_location=self.device)
                    if self.model.stacking_enabled:
                        # Warm-start a STACKED (132-dim) net from a non-stacked (44-dim) policy:
                        # tile the first-layer weight across the STACK_DEPTH frame slots, scaled
                        # 1/depth. The padded frames are identical at reset, so the stacked net
                        # initially reproduces the single-frame policy (~42%), then refines with
                        # motion info -> avoids the dead-slow from-scratch convergence (exp018).
                        depth = self.model.stack_depth
                        model_sd = net.state_dict()
                        adapted = {}
                        for k, v in sd.items():
                            if k in model_sd and model_sd[k].shape != v.shape \
                               and v.dim() == 2 and model_sd[k].shape[1] == v.shape[1] * depth:
                                adapted[k] = v.repeat(1, depth) / depth
                            else:
                                adapted[k] = v
                        net.load_state_dict(adapted)
                    else:
                        net.load_state_dict(sd)
                self.total_steps = self.observe_steps   # skip random-observe -> use the loaded policy from step 1
                # but DON'T start gradient updates until the buffer holds DRL_WARMSTART_FILL
                # transitions collected BY THE LOADED POLICY. Training immediately on a near-empty
                # buffer feeds correlated mini-batches that corrupt the warm-started actor into a
                # degenerate spin/reverse policy (the exp008 collapse). Seeding a diverse buffer first
                # makes the warm-start stable.
                self.train_start_len = int(os.environ.get('DRL_WARMSTART_FILL', '4000'))
                print(f"[warmstart] loaded best weights from {warmstart}; fresh buffer; "
                      f"seeding {self.train_start_len} steps with loaded policy before training", flush=True)
            self.sm.store_model(self.model)

        # Re-apply n-step config to the model unconditionally: a RESUMED model is unpickled
        # from a checkpoint that may predate n-step, so these attrs can be missing. Fresh
        # models already set them in OffPolicyAgent.__init__; this makes resume safe too.
        self.model.n_step = N_STEP
        self.model.nstep_discount = self.model.discount_factor ** N_STEP

        self.graph.session_dir = self.sm.session_dir
        self.logger = Logger(self.training, self.sm.machine_dir, self.sm.session_dir, self.sm.session,
                             self.model.get_model_parameters(), self.model.get_model_configuration(),
                             str(util.stage), self.algorithm, self.episode)
        # Universal training harness: MA100-success + reward-component metrics -> _metrics.tsv
        # + training.png. Training runs only (test/eval use the logger's test results).
        self.metrics = TrainingMetrics(self.sm.session_dir) if self.training else None

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
            comp_sums = [0.0] * len(REWARD_COMPONENT_NAMES)   # per-episode reward-component sums
            action_past = [0.0, 0.0]
            state = util.init_episode(self)
            if self.stacking:
                state = self.frame_stacker.reset(state)
            nstep_queue = deque()          # holds the last <=n raw transitions for n-step accumulation

            util.unpause_simulation(self, self.real_robot)
            time.sleep(0.5)
            episode_start = time.perf_counter()

            while not episode_done:
                # Random actions during the observe phase to seed the buffer with
                # diverse data; greedy actor + OU noise afterwards.
                if self.training and self.total_steps < self.observe_steps:
                    action = self.model.get_action_random()
                else:
                    action = self.model.get_action(state, self.training, self.total_steps + step)  # GLOBAL step -> OU sigma decays over training

                next_state, reward, episode_done, outcome, distance_traveled, reward_components, initial_distance = util.step(self, action, action_past)
                if self.stacking:
                    next_state = self.frame_stacker.push(next_state)
                action_past = copy.deepcopy(action)
                reward_sum += reward
                for i, c in enumerate(reward_components[:len(comp_sums)]):
                    comp_sums[i] += c

                if self.training:
                    # N-step: buffer raw steps; emit a transition whose reward is the
                    # gamma-discounted n-step sum and whose next_state is s_{t+n}. n=1 ->
                    # emits every step immediately = exactly the old 1-step behaviour.
                    nstep_queue.append((state, action, reward, next_state, episode_done))
                    if len(nstep_queue) >= self.model.n_step:
                        self._emit_nstep(nstep_queue, self.model.n_step)
                    if episode_done:                       # drain the < n tail at episode end
                        while nstep_queue:
                            self._emit_nstep(nstep_queue, len(nstep_queue))
                    if self.replay_buffer.get_length() >= self.train_start_len:
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
            self.finish_episode(step, duration, outcome, distance_traveled, reward_sum, loss_critic, loss_actor, comp_sums, initial_distance)

            # Experiment episode budget (env: DRL_MAX_EPISODES). Clean stop for "train N eps then analyze".
            if MAX_TRAINING_EPISODES and self.training and self.episode >= MAX_TRAINING_EPISODES:
                print(f"[done] reached MAX_TRAINING_EPISODES={MAX_TRAINING_EPISODES} — saving + stopping.", flush=True)
                self.sm.save_session(self.episode, self.model.networks, self.graph.graphdata, self.replay_buffer.buffer)
                self.graph.draw_plots(self.episode)
                sys.exit(0)

            # Central native evaluation (env: DRL_TEST_EPISODES). After N test episodes, print
            # the standard summary and exit cleanly — this is `scripts/eval.sh` under the hood.
            # Use the logger's test_entry (true count); self.episode starts at the loaded episode.
            if TEST_EPISODES and not self.training and self.logger.test_entry >= TEST_EPISODES:
                from ..evaluation.evaluate import summarize
                self.logger.file_log.flush()
                results_dir = os.path.join(os.environ.get('DRLNAV_BASE_PATH', '.'),
                                           'src/turtlebot3_drl/turtlebot3_drl/evaluation/results')
                summarize(self.logger.file_log.name, f"{self.algorithm} {self.load_session} ep{self.load_episode}",
                          results_dir)
                sys.exit(0)

    def _emit_nstep(self, queue, horizon):
        """Store ONE n-step transition for the head of `queue`, then pop the head.
          reward  = sum_{k=0}^{horizon-1} gamma^k * r_{t+k}
          next_s  = next_state of the last step in the window (s_{t+horizon})
          done    = True iff a step in the window terminated the episode
        Episodes here always end on a terminal/timeout, so a short tail (horizon<n) always
        has done=True -> the gamma**n bootstrap is masked out in train(); only the reward
        sum matters. n=1 reproduces the original 1-step transition exactly."""
        gamma = self.model.discount_factor
        s0, a0 = queue[0][0], queue[0][1]
        R, discount, done_n = 0.0, 1.0, False
        last_next_state = queue[0][3]
        for k in range(horizon):
            _, _, r_k, ns_k, d_k = queue[k]
            R += discount * r_k
            discount *= gamma
            last_next_state = ns_k
            if d_k:
                done_n = True
                break
        self.replay_buffer.add_sample(s0, a0, [R], last_next_state, [done_n])
        queue.popleft()

    def finish_episode(self, step, eps_duration, outcome, dist_traveled, reward_sum, loss_critic, loss_actor, comp_sums=None, initial_distance=0.0):
        if self.total_steps < self.observe_steps:
            print(f"Observe phase: {self.total_steps}/{self.observe_steps} steps")
            return

        self.episode += 1

        if not self.training:
            print(f"Epi: {self.episode:<5}R: {reward_sum:<8.0f}outcome: {util.translate_outcome(outcome):<13}"
                  f"steps: {step:<6}steps_total: {self.total_steps:<7}time: {eps_duration:<6.2f}")
            self.logger.update_test_results(step, outcome, dist_traveled, eps_duration, 0, initial_distance)
            return

        # Universal harness: record metrics (-> _metrics.tsv) and print the live line w/ MA100.
        ma100 = self.metrics.record(self.episode, self.total_steps, outcome, reward_sum,
                                    loss_critic / step, loss_actor / step,
                                    comp_sums or [0.0] * len(REWARD_COMPONENT_NAMES))
        print(episode_line(self.episode, util.translate_outcome(outcome), step,
                           self.total_steps, eps_duration, ma100), flush=True)

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
            plots.draw(self.sm.session_dir)

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
            if self.stacking:
                state = self.frame_stacker.reset(state)
            util.unpause_simulation(self, self.real_robot)
            time.sleep(0.5)
            outcome = 0
            while not episode_done:
                action = self.model.get_action(state, False, step)  # is_training=False -> greedy, no noise
                next_state, _, episode_done, outcome, _, _, _ = util.step(self, action, action_past)
                if self.stacking:
                    next_state = self.frame_stacker.push(next_state)
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
