import os
# ===================================================================== #
#                           GENERAL SETTINGS                            #
# ===================================================================== #
# Experiment-overridable knobs (set by scripts/run_experiment.sh per run, so each
# experiment is reproducible without editing/rebuilding). Defaults below apply when
# the env vars are unset.

ENABLE_BACKWARD          = False    # Forward-only action space [0, 0.22] m/s. The 100% DDPG example used False; backward let the robot reverse into walls (~80% wall collisions).
ENABLE_STACKING          = False    # Enable processing multiple consecutive scan frames at every observation step
ENABLE_VISUAL            = False    # Meant to be used only during evaluation/testing phase
ENABLE_TRUE_RANDOM_GOALS = False    # If false, goals are selected semi-randomly from a list of known valid goal positions
ENABLE_DYNAMIC_GOALS     = (os.environ.get('DRL_DYNAMIC_GOALS', 'True') == 'True')   # curriculum on/off (env: DRL_DYNAMIC_GOALS). Goal spawns on a difficulty_radius ring, clamped to [CURRICULUM_MIN_RADIUS, CURRICULUM_MAX_RADIUS].
MAX_TRAINING_EPISODES    = int(os.environ.get('DRL_MAX_EPISODES', '0'))              # stop training after N episodes (env: DRL_MAX_EPISODES); 0 = unlimited.
CURRICULUM_MIN_RADIUS    = 2.0      # FIX: floor so goals are never trivial. Old 0.5 collapsed to ~0.2m goals -> 85% on a trivial task but 27% on the real benchmark.
CURRICULUM_MAX_RADIUS    = 4.0      # cap; benchmark static goals reach ~3.4m, so [2.0,4.0] spans real-navigation difficulty.
MODEL_STORE_INTERVAL     = 100      # Store the model weights every N episodes
GRAPH_DRAW_INTERVAL      = 10       # Draw the graph every N episodes (drawing too often will slow down training)
GRAPH_AVERAGE_REWARD     = 10       # Average the reward graph over every N episodes


# ===================================================================== #
#                         ENVIRONMENT SETTINGS                          #
# ===================================================================== #

# --- SIMULATION ENVIRONMENT SETTINGS ---
REWARD_FUNCTION = os.environ.get('DRL_REWARD', 'A')   # env: DRL_REWARD (S=clean sparse, A/B/..=dense). Defined in reward.py
EPISODE_TIMEOUT_SECONDS = 50    # Number of seconds after which episode timeout occurs

TOPIC_SCAN = 'scan'
TOPIC_VELO = 'cmd_vel'
TOPIC_ODOM = 'odom'

EPISODE_TIMEOUT_SECONDS     = 50    # Number of seconds after which episode timeout occurs
ARENA_LENGTH                = 4.2   # meters
ARENA_WIDTH                 = 4.2   # meters
SPEED_LINEAR_MAX            = 0.22  # m/s
SPEED_ANGULAR_MAX           = 2.0   # rad/s

LIDAR_DISTANCE_CAP          = 3.5   # meters
THRESHOLD_COLLISION         = 0.13  # meters
THREHSOLD_GOAL              = 0.20  # meters

# New stage-9 world runs real_time_factor=0 (uncapped, wall-clock paced). Sim-speed
# scaling is no longer parsed from a vendored SDF (R3) — it's this constant.
SIM_SPEED                   = 1

OBSTACLE_RADIUS             = 0.16  # meters
MAX_NUMBER_OBSTACLES        = 6
ENABLE_MOTOR_NOISE          = False # Add normally distributed noise to motor output to simulate hardware imperfections

# --- REAL ROBOT ENVIRONMENT SETTINGS ---
REAL_TOPIC_SCAN  = 'scan'
REAL_TOPIC_VELO  = 'cmd_vel'
REAL_TOPIC_ODOM  = 'odom'

# LiDAR density count your robot is providing
# NOTE: If you change this value you also have to modify
# NUM_SCAN_SAMPLES for the model in drl_environment.py
# e.g. if you increase this by 320 samples also increase
# NUM_SCAN_SAMPLES by 320 samples.
REAL_N_SCAN_SAMPLES         = 40

REAL_ARENA_LENGTH           = 4.2   # meters
REAL_ARENA_WIDTH            = 4.2   # meters
REAL_SPEED_LINEAR_MAX       = 0.22  # in m/s
REAL_SPEED_ANGULAR_MAX      = 2.0   # in rad/s

REAL_LIDAR_CORRECTION       = 0.40  # meters, subtracted from the real LiDAR values
REAL_LIDAR_DISTANCE_CAP     = 3.5   # meters, scan distances are capped this value
REAL_THRESHOLD_COLLISION    = 0.11  # meters, minimum distance to an object that counts as a collision
REAL_THRESHOLD_GOAL         = 0.35  # meters, minimum distance to goal that counts as reaching the goal


# ===================================================================== #
#                       DRL ALGORITHM SETTINGS                          #
# ===================================================================== #

# DRL parameters
REWARD_FUNCTION = os.environ.get('DRL_REWARD', 'A')   # env: DRL_REWARD (S=clean sparse, P=sparse+progress, A/B=dense). Defined in reward.py
REWARD_SCALE    = 1.0       # 1.0 = canonical DDPG/TD3 magnitude (lean repo starts with DDPG). Set 0.1 for SAC's entropy/Q balance. Scales reward MAGNITUDE only — compare success rate, not raw reward, across runs.
PROGRESS_K      = float(os.environ.get('DRL_PROGRESS_K', '2.0'))   # reward_P: gain on closing-distance shaping (potential-based, policy-invariant). env: DRL_PROGRESS_K.
ACTION_SIZE     = 2         # Not used for DQN, see DQN_ACTION_SIZE
HIDDEN_SIZE     = 512       # 1024 was tried for SAC v2 — caused policy collapse, reverted

BATCH_SIZE      = 128       # Reference winning-DDPG value (tomasvr/turtlebot3_drlnav). 1024 + low lr starved effective updates -> policy collapse on stage 9.
BUFFER_SIZE     = 1000000   # Number of samples stored in replay buffer before FIFO
DISCOUNT_FACTOR = 0.99
LEARNING_RATE   = 0.003     # Reference winning-DDPG value. 3e-4 was 10x too slow: actor locked in the r_vlinear max-speed habit before the critic learned obstacle avoidance -> collapse.
TAU             = 0.003     # Reference winning-DDPG value (NOT a "SAC value"). 3e-4 made target tracking 10x too slow.

OBSERVE_STEPS   = 25000     # At training start random actions are taken for N steps for better exploration
STEP_TIME       = 0.01      # Reference winning-DDPG value: a defined, consistent action-execution window. 0.0 made the MDP timestep equal to variable ROS service latency (ill-defined).
EPSILON_DECAY   = 0.9995    # Epsilon decay per step
EPSILON_MINIMUM = 0.05

# Eval-based checkpoint selection: every MODEL_STORE_INTERVAL training eps,
# pause and run N deterministic-policy episodes; save `_best.pt` only when
# this score improves. Set to 0 to disable.
VAL_EPS_PER_CHECKPOINT   = 20    # every MODEL_STORE_INTERVAL eps, pause and run N deterministic (no-noise) episodes; logs to _eval_stage<N>.tsv in the session dir and saves *_best.pt on improvement. 0 disables.

# Stacking
STACK_DEPTH = 3             # Number of subsequent frames processed per step
FRAME_SKIP  = 4             # Number of frames skipped in between subsequent frames

# Episode outcome enumeration
UNKNOWN = 0
SUCCESS = 1
COLLISION_WALL = 2
COLLISION_OBSTACLE = 3
TIMEOUT = 4
TUMBLE = 5
RESULTS_NUM = 6