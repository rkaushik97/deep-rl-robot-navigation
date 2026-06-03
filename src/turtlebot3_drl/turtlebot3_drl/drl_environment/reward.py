from ..common.settings import REWARD_FUNCTION, REWARD_SCALE, COLLISION_OBSTACLE, COLLISION_WALL, TUMBLE, SUCCESS, TIMEOUT, RESULTS_NUM

goal_dist_initial = 0

reward_function_internal = None

# Last-call component breakdown for the debug publisher. The env reads this
# after get_reward() — keeps get_reward()'s scalar return type intact.
last_components = {}

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

# Define your own reward function by defining a new function: 'get_reward_X'
# Replace X with your reward function name and configure it in settings.py

def reward_initalize(init_distance_to_goal):
    global goal_dist_initial
    goal_dist_initial = init_distance_to_goal

function_name = "get_reward_" + REWARD_FUNCTION
reward_function_internal = globals()[function_name]
if reward_function_internal == None:
    quit(f"Error: reward function {function_name} does not exist")
