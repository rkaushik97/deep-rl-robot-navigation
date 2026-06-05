from ..common.settings import REWARD_FUNCTION, REWARD_SCALE, COLLISION_OBSTACLE, COLLISION_WALL, TUMBLE, SUCCESS, TIMEOUT, RESULTS_NUM, PROGRESS_K, OBSTACLE_K, OBSTACLE_SAFE, SPEED_LINEAR_MAX, SPEED_ANGULAR_MAX

goal_dist_initial = 0
prev_goal_dist = 0      # for potential-based progress shaping (reward_P); reset per episode
prev_min_obs = 99.0     # for approach-aware obstacle penalty (reward_O); reset per episode
prev_action_linear = 0.0   # for control-smoothness penalty (reward_VH); reset per episode
prev_action_angular = 0.0  # for control-smoothness penalty (reward_VH); reset per episode

reward_function_internal = None

# Last-call component breakdown. The env reads this after get_reward() and ships
# it (in REWARD_COMPONENT_NAMES order) so the learner can report which shaping
# term dominates — keeps get_reward()'s scalar return type intact.
last_components = {}

# Canonical order shared by env -> actor -> learner -> metrics file.
REWARD_COMPONENT_NAMES = ['r_yaw', 'r_distance', 'r_obstacle', 'r_vlinear', 'r_vangular', 'r_step', 'r_terminal']


def get_last_components_ordered():
    return [float(last_components.get(k, 0.0)) for k in REWARD_COMPONENT_NAMES]

def get_reward(succeed, action_linear, action_angular, distance_to_goal, goal_angle, min_obstacle_distance):
    # SAC needs |reward| ~ O(10s) so the entropy term in the actor loss is
    # the same order of magnitude as the Q values. Pure scaling — relative
    # shaping is unchanged; set REWARD_SCALE=1.0 in settings.py for the
    # original TD3/DDPG/REDQ training behaviour.
    return reward_function_internal(succeed, action_linear, action_angular, distance_to_goal, goal_angle, min_obstacle_distance) * REWARD_SCALE

def get_reward_A(succeed, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist):
        # [-3.14, 0]
        r_yaw = -1 * abs(goal_angle)

        # [-4, 0]
        r_vangular = -1 * (action_angular**2)

        # [-1, 1]
        r_distance = (2 * goal_dist_initial) / (goal_dist_initial + goal_dist) - 1

        # [-20, 0]
        if min_obstacle_dist < 0.22:
            r_obstacle = -20
        else:
            r_obstacle = 0

        # [-2 * (2.2^2), 0]
        r_vlinear = -1 * (((0.22 - action_linear) * 10) ** 2)

        reward = r_yaw + r_distance + r_obstacle + r_vlinear + r_vangular - 1

        terminal_bonus = 0.0
        if succeed == SUCCESS:
            terminal_bonus = 2500
        elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL:
            terminal_bonus = -2000
        reward += terminal_bonus

        last_components.clear()
        last_components.update({
            'r_yaw': float(r_yaw),
            'r_distance': float(r_distance),
            'r_obstacle': float(r_obstacle),
            'r_vlinear': float(r_vlinear),
            'r_vangular': float(r_vangular),
            'r_step': -1.0,
            'r_terminal': float(terminal_bonus),
        })
        return float(reward)

def get_reward_B(succeed, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist):
        # SPARSE reward: no directional shaping. The network figures out *how* to
        # navigate from only (a) a per-step time penalty and (b) a terminal
        # goal/collision signal. Relies on ENABLE_DYNAMIC_GOALS so the goal is
        # reachable by chance early on (close-ring curriculum) -> the agent
        # actually sees the +success signal to learn from.
        #
        # Why a step penalty and not pure terminal-only: without it, "freeze /
        # spin in place" is a safe local optimum (reward 0 beats risking -collision).
        # The -1/step makes idling cost ~ -timeout_steps, so the ordering is
        # success (>0) >> timeout (mildly negative) > collision (very negative),
        # which forces goal-seeking instead of bleeding out.
        r_step = -1.0

        terminal_bonus = 0.0
        if succeed == SUCCESS:
            terminal_bonus = 2500.0
        elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL:
            terminal_bonus = -2000.0

        reward = r_step + terminal_bonus

        last_components.clear()
        last_components.update({
            'r_yaw': 0.0,
            'r_distance': 0.0,
            'r_obstacle': 0.0,
            'r_vlinear': 0.0,
            'r_vangular': 0.0,
            'r_step': float(r_step),
            'r_terminal': float(terminal_bonus),
        })
        return float(reward)

def get_reward_S(succeed, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist):
        # CLEAN SPARSE reward — the unhackable baseline. Only signal: +1 reaching the
        # goal, -1 for any collision, 0 every other step. Time pressure comes from the
        # discount (gamma<1): reaching sooner is worth more, and "freeze -> timeout"
        # returns 0, which loses to any real success. Q stays in [-1, 1] -> extremely
        # well conditioned (vs the +/-2500 that blew Q up). If the agent learns to
        # navigate under THIS, the algorithm genuinely works; only then add dense terms.
        terminal = 0.0
        if succeed == SUCCESS:
            terminal = 1.0
        elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL:
            terminal = -1.0

        last_components.clear()
        last_components.update({
            'r_yaw': 0.0, 'r_distance': 0.0, 'r_obstacle': 0.0,
            'r_vlinear': 0.0, 'r_vangular': 0.0, 'r_step': 0.0,
            'r_terminal': float(terminal),
        })
        return float(terminal)

def get_reward_P(succeed, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist):
        # SPARSE + POTENTIAL-BASED PROGRESS (exp002). One addition on top of the clean
        # sparse baseline: a dense reward for *closing distance* to the goal.
        #   r_progress = K * (prev_dist - curr_dist)   (clipped to +/-1)
        # This is potential-based shaping (Phi = -distance; Ng et al. 1999): it is
        # PROVABLY policy-invariant, so it speeds learning without changing the optimal
        # policy and CANNOT be reward-hacked (unlike r_vlinear). It supplies the gradient
        # toward the goal that pure sparse lacked (exp001 = 0% success / never reached a
        # goal). Crashing ends the episode early -> forfeits future progress reward, so
        # collision-avoidance is incentivised implicitly; standing still earns 0, so there
        # is no survive-to-timeout trap.
        global prev_goal_dist
        r_progress = max(-1.0, min(1.0, PROGRESS_K * (prev_goal_dist - goal_dist)))
        prev_goal_dist = goal_dist

        terminal = 0.0
        if succeed == SUCCESS:
            terminal = 1.0
        elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL:
            terminal = -1.0

        reward = r_progress + terminal
        last_components.clear()
        last_components.update({
            'r_yaw': 0.0, 'r_distance': float(r_progress), 'r_obstacle': 0.0,
            'r_vlinear': 0.0, 'r_vangular': 0.0, 'r_step': 0.0,
            'r_terminal': float(terminal),
        })
        return float(reward)

def get_reward_O(succeed, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist):
        # exp003: sparse + progress + smooth OBSTACLE-PROXIMITY penalty.
        # Failure analysis: ~74% of wall collisions are "navigate toward goal at near-max
        # speed, then clip a wall" with no slowdown near obstacles. This term makes being
        # near an obstacle costly: 0 at >= OBSTACLE_SAFE m, ramping linearly to -OBSTACLE_K
        # at contact -> teaches clearance AND implicit slowdown in tight spaces. Kept small
        # (~0.4 peak, comparable to the progress term) so Q stays O(1).
        global prev_goal_dist, prev_min_obs
        r_progress = max(-1.0, min(1.0, PROGRESS_K * (prev_goal_dist - goal_dist)))
        prev_goal_dist = goal_dist

        # APPROACH-AWARE obstacle penalty: fire ONLY when CLOSING on an obstacle inside the
        # danger zone (quadratic in proximity). A static proximity penalty (every near-wall
        # step) accumulated to ~-8/episode and COLLAPSED the policy, because navigating a
        # cluttered stage REQUIRES being near walls -> the penalty punished necessary
        # navigation, the value function craters, the agent over-avoids (timeouts) and still
        # crashes. Gating on "getting closer" means it fires only while driving TOWARD a wall
        # (the actual clip failure), never while following a wall in parallel -> can't
        # accumulate, targets the real behaviour.
        r_obstacle = 0.0
        if min_obstacle_dist < OBSTACLE_SAFE and min_obstacle_dist < prev_min_obs:
            prox = (OBSTACLE_SAFE - min_obstacle_dist) / OBSTACLE_SAFE   # 0 at SAFE, 1 at contact
            r_obstacle = -OBSTACLE_K * prox * prox
        prev_min_obs = min_obstacle_dist

        terminal = 0.0
        if succeed == SUCCESS:
            terminal = 1.0
        elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL:
            terminal = -1.0

        reward = r_progress + r_obstacle + terminal
        last_components.clear()
        last_components.update({
            'r_yaw': 0.0, 'r_distance': float(r_progress), 'r_obstacle': float(r_obstacle),
            'r_vlinear': 0.0, 'r_vangular': 0.0, 'r_step': 0.0,
            'r_terminal': float(terminal),
        })
        return float(reward)

def get_reward_V(succeed, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist):
        # exp004: sparse + progress + SPEED-MODULATED obstacle penalty.
        # Failure analysis: ~85% of walls are "clip while navigating", 56% at high forward
        # speed. The proximity penalty (exp003) was neutral because navigating a cluttered
        # stage REQUIRES being near walls. This instead penalises being FAST while near an
        # obstacle -> teaches "slow down in clutter" without punishing slow careful gap-
        # navigation (penalty is ~0 when slow OR far; max only when fast AND close).
        global prev_goal_dist
        r_progress = max(-1.0, min(1.0, PROGRESS_K * (prev_goal_dist - goal_dist)))
        prev_goal_dist = goal_dist

        r_speed = 0.0
        if min_obstacle_dist < OBSTACLE_SAFE:
            speed = max(0.0, action_linear) / SPEED_LINEAR_MAX        # 0..1 forward-speed fraction
            prox = (OBSTACLE_SAFE - min_obstacle_dist) / OBSTACLE_SAFE  # 0 at SAFE, 1 at contact
            r_speed = -OBSTACLE_K * speed * prox

        terminal = 0.0
        if succeed == SUCCESS:
            terminal = 1.0
        elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL:
            terminal = -1.0

        reward = r_progress + r_speed + terminal
        last_components.clear()
        last_components.update({
            'r_yaw': 0.0, 'r_distance': float(r_progress), 'r_obstacle': float(r_speed),
            'r_vlinear': 0.0, 'r_vangular': 0.0, 'r_step': 0.0, 'r_terminal': float(terminal),
        })
        return float(reward)

def get_reward_VH(succeed, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist):
        # exp007: progress + speed-modulated obstacle penalty (reward_V) + GATED heading +
        # control smoothness. Targets the dominant remaining failure (mid-journey clip-while-
        # navigating, ~72-80% of walls in exp005/exp006) without re-introducing the over-caution
        # collapse: heading is PENALTY-ONLY and GATED to open space (>= OBSTACLE_SAFE), so it
        # never reinforces driving straight into a wall; smoothness damps the lin=-1 reversing
        # over-caution seen in exp006's clean eval. action_linear/angular are physical units.
        global prev_goal_dist, prev_action_linear, prev_action_angular
        r_progress = max(-1.0, min(1.0, PROGRESS_K * (prev_goal_dist - goal_dist)))
        prev_goal_dist = goal_dist

        r_speed = 0.0
        if min_obstacle_dist < OBSTACLE_SAFE:
            speed = max(0.0, action_linear) / SPEED_LINEAR_MAX
            prox = (OBSTACLE_SAFE - min_obstacle_dist) / OBSTACLE_SAFE
            r_speed = -OBSTACLE_K * speed * prox

        r_heading = 0.0
        if min_obstacle_dist >= OBSTACLE_SAFE:            # gated: only steer toward goal in open space
            r_heading = -0.10 * (abs(goal_angle) / 3.14159265)

        d_ang = abs(action_angular - prev_action_angular) / (2.0 * SPEED_ANGULAR_MAX)
        d_lin = abs(action_linear - prev_action_linear) / SPEED_LINEAR_MAX
        r_smooth = -0.10 * (d_ang + 0.25 * d_lin)
        prev_action_linear = action_linear
        prev_action_angular = action_angular

        terminal = 1.0 if succeed == SUCCESS else (-1.0 if succeed in (COLLISION_OBSTACLE, COLLISION_WALL) else 0.0)
        reward = r_progress + r_speed + r_heading + r_smooth + terminal
        last_components.clear()
        last_components.update({
            'r_yaw': float(r_heading), 'r_distance': float(r_progress), 'r_obstacle': float(r_speed),
            'r_vlinear': float(r_smooth), 'r_vangular': 0.0, 'r_step': 0.0, 'r_terminal': float(terminal),
        })
        return float(reward)

# Define your own reward function by defining a new function: 'get_reward_X'
# Replace X with your reward function name and configure it in settings.py

def reward_initalize(init_distance_to_goal):
    global goal_dist_initial, prev_goal_dist, prev_min_obs, prev_action_linear, prev_action_angular
    goal_dist_initial = init_distance_to_goal
    prev_goal_dist = init_distance_to_goal
    prev_min_obs = 99.0
    prev_action_linear = 0.0
    prev_action_angular = 0.0

function_name = "get_reward_" + REWARD_FUNCTION
reward_function_internal = globals()[function_name]
if reward_function_internal == None:
    quit(f"Error: reward function {function_name} does not exist")
