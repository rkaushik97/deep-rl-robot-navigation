#!/usr/bin/env python3
# Deterministic replay of a saved actor + per-collision CAUSE classification.
#   python3 scripts/replay_analyze.py <actor_ckpt.pt> [n_eps]
# Runs the greedy policy, and for every wall collision records the state AT collision
# (goal distance/angle, nearest-obstacle distance, action) to attribute the cause.
import copy, sys, time, math
import torch
import rclpy
from rclpy.node import Node
from turtlebot3_msgs.srv import DrlStep, Goal
from ros_gz_interfaces.srv import ControlWorld
from turtlebot3_drl.common import utilities as util
from turtlebot3_drl.algorithms.ddpg.ddpg import DDPG
from turtlebot3_drl.common.settings import SUCCESS, COLLISION_WALL, LIDAR_DISTANCE_CAP, ENABLE_STACKING, STACK_DEPTH, FRAME_SKIP
from turtlebot3_drl.drl_environment.drl_environment import NUM_SCAN_SAMPLES, MAX_GOAL_DISTANCE
from turtlebot3_drl.drl_agent.drl_agent import FrameStacker


def parse_state(s):
    n = NUM_SCAN_SAMPLES
    min_obs = min(s[0:n]) * LIDAR_DISTANCE_CAP
    gd = s[n] * MAX_GOAL_DISTANCE
    ga = math.degrees(s[n+1] * math.pi)
    return gd, ga, min_obs


class Replay(Node):
    def __init__(self, ckpt, n_eps, algo='ddpg'):
        super().__init__('replay_analyze')
        self.real_robot = 0
        self.device = torch.device('cpu')
        # DDPG/TD3 share the deterministic Actor (fa1/fa2/fa3); SAC uses a GaussianActor.
        # get_action(state, False, step) is deterministic for all three (SAC -> tanh(mu)).
        if algo == 'sac':
            from turtlebot3_drl.algorithms.sac.sac import SAC
            self.model = SAC(self.device, 1)
        else:
            self.model = DDPG(self.device, 1)
        self.model.actor.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.actor.eval()
        self.step_comm_client = self.create_client(DrlStep, 'step_comm')
        self.goal_comm_client = self.create_client(Goal, 'goal_comm')
        self.gazebo_control = self.create_client(ControlWorld, f'/world/drl_stage{util.stage}/control')
        self.run(n_eps)

    def run(self, n_eps):
        util.pause_simulation(self, 0)
        rec = []
        stacker = FrameStacker(STACK_DEPTH, FRAME_SKIP) if ENABLE_STACKING else None
        for ep in range(n_eps):
            util.wait_new_goal(self)
            raw = util.init_episode(self)
            state = stacker.reset(raw) if stacker else raw   # stacked 132-dim if DRL_STACKING=1
            ap = [0.0, 0.0]
            util.unpause_simulation(self, 0); time.sleep(0.5)
            done = False; step = 0; outcome = 0; last = (0, 0, 0, 0, 0)
            while not done:
                action = self.model.get_action(state, False, step)   # greedy, no noise
                ns, _, done, outcome, _, _ = util.step(self, action, ap)
                gd, ga, mo = parse_state(ns)                          # parse the RAW frame for cause classification
                last = (gd, ga, mo, action[0], action[1])
                ap = copy.deepcopy(action)
                state = stacker.push(ns) if stacker else copy.deepcopy(ns)
                step += 1
                time.sleep(self.model.step_time)
            util.pause_simulation(self, 0)
            rec.append((outcome, step, last))
            print(f"[{ep+1}/{n_eps}] {util.translate_outcome(outcome):10s} steps={step:<4} "
                  f"gd={last[0]:.2f} ga={last[1]:.0f}deg minObs={last[2]:.2f} lin={last[3]:.2f}", flush=True)
        self.classify(rec)

    def classify(self, rec):
        n = len(rec)
        succ = sum(1 for r in rec if r[0] == SUCCESS)
        walls = [r for r in rec if r[0] == COLLISION_WALL]
        nw = len(walls)
        c = {'early/orientation (<40 steps)': 0,
             'wrong-heading (goal >100deg off-axis)': 0,
             'CLIP-WHILE-NAVIGATING (the problem)': 0}
        near_goal = mid = fast = 0
        for o, step, last in walls:
            gd, ga, mo, lin, ang = last
            if step < 40:
                c['early/orientation (<40 steps)'] += 1
            elif abs(ga) > 100:
                c['wrong-heading (goal >100deg off-axis)'] += 1
            else:
                c['CLIP-WHILE-NAVIGATING (the problem)'] += 1
                if gd < 0.5: near_goal += 1
                else: mid += 1
                if lin > 0.5: fast += 1
        pct = lambda v: f"{100*v/max(1,nw):.0f}%"
        print("\n==== REPLAY CAUSE BREAKDOWN ====")
        print(f"episodes={n}  success={succ} ({100*succ/max(1,n):.0f}%)  wall_collisions={nw} ({100*nw/max(1,n):.0f}%)")
        for k, v in c.items():
            print(f"  {k:42s} {v:3d}  ({pct(v)} of walls)")
        if nw:
            print(f"    - of clip-while-navigating: near-goal(<0.5m)={near_goal}, mid-journey={mid}")
            print(f"    - clip happened at high speed (lin>0.5): {fast} ({pct(fast)} of walls)")


def main():
    import os
    ck = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    algo = (sys.argv[3] if len(sys.argv) > 3 else os.environ.get('DRL_EVAL_ALGO', 'ddpg')).lower()
    # Guard: deterministic eval MUST run on the fixed benchmark (DYNAMIC_GOALS=False).
    # With the curriculum ON the goal difficulty adapts to the policy -> moving, self-
    # penalizing benchmark (reads ~20-25pp too low). Use scripts/clean_eval.sh which sets it.
    from turtlebot3_drl.common.settings import ENABLE_DYNAMIC_GOALS
    if ENABLE_DYNAMIC_GOALS:
        print("\n*** WARNING: ENABLE_DYNAMIC_GOALS=True — eval is on the ADAPTING curriculum, "
              "NOT the fixed benchmark. Numbers will be too low. Launch the env with "
              "DRL_DYNAMIC_GOALS=False (use scripts/clean_eval.sh). ***\n", flush=True)
    else:
        print("[clean_eval] DYNAMIC_GOALS=False — fixed benchmark goals (correct).", flush=True)
    print(f"[clean_eval] algorithm = {algo}", flush=True)
    rclpy.init(); Replay(ck, n, algo); rclpy.shutdown()


if __name__ == '__main__':
    main()
