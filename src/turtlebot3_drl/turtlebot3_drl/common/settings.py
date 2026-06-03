# ===================================================================== #
#                           GENERAL SETTINGS                            #
# ===================================================================== #

ENABLE_BACKWARD          = True     # Enable backward movement of the robot (matches TD3's action space; lets robot reverse out of obstacle traps on stage 9)
ENABLE_STACKING          = False    # Enable processing multiple consecutive scan frames at every observation step
ENABLE_VISUAL            = False    # Meant to be used only during evaluation/testing phase
ENABLE_TRUE_RANDOM_GOALS = False    # If false, goals are selected semi-randomly from a list of known valid goal positions
ENABLE_DYNAMIC_GOALS     = False    # If true, goal difficulty (distance) is adapted according to current success rate
MODEL_STORE_INTERVAL     = 100      # Store the model weights every N episodes
GRAPH_DRAW_INTERVAL      = 10       # Draw the graph every N episodes (drawing too often will slow down training)
GRAPH_AVERAGE_REWARD     = 10       # Average the reward graph over every N episodes


# ===================================================================== #
#                         ENVIRONMENT SETTINGS                          #
# ===================================================================== #

# --- SIMULATION ENVIRONMENT SETTINGS ---
REWARD_FUNCTION = "A"           # Defined in reward.py
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
REWARD_FUNCTION = "A"       # Defined in reward.py
REWARD_SCALE    = 1.0       # 1.0 = canonical DDPG/TD3 magnitude (lean repo starts with DDPG). Set 0.1 for SAC's entropy/Q balance. Scales reward MAGNITUDE only — compare success rate, not raw reward, across runs.
ACTION_SIZE     = 2         # Not used for DQN, see DQN_ACTION_SIZE
HIDDEN_SIZE     = 512       # 1024 was tried for SAC v2 — caused policy collapse, reverted

BATCH_SIZE      = 256       # Steadier SAC critic updates (sac_5 value)
BUFFER_SIZE     = 1000000   # Number of samples stored in replay buffer before FIFO
DISCOUNT_FACTOR = 0.99
LEARNING_RATE   = 0.0003    # canonical/stable SAC lr (3e-4); 3e-3 risks entropy/Q blow-up
TAU             = 0.003

OBSERVE_STEPS   = 25000     # At training start random actions are taken for N steps for better exploration
STEP_TIME       = 0.05      # Deadline-paced ~20Hz control cadence (sac_5 value). 0.0 (uncapped) inflated episodes to ~450 steps / 45% timeouts. gz-sim is wall-clock paced.
EPSILON_DECAY   = 0.9995    # Epsilon decay per step
EPSILON_MINIMUM = 0.05

# DQN parameters
DQN_ACTION_SIZE = 5
TARGET_UPDATE_FREQUENCY = 1000

# DDPG parameters

# TD3 parameters
POLICY_NOISE            = 0.2
POLICY_NOISE_CLIP       = 0.5
POLICY_UPDATE_FREQUENCY = 2

# REDQ parameters
REDQ_ENSEMBLE_SIZE      = 10     # N — critics in the ensemble
REDQ_TARGET_SUBSET      = 2      # M — random subset used when building the TD target (M <= N)
REDQ_UTD_RATIO          = 20     # G — gradient updates per env step
REDQ_INIT_TEMPERATURE   = 0.2    # initial entropy coefficient α
REDQ_AUTOTUNE_ALPHA     = True   # learn α so policy entropy tracks REDQ_TARGET_ENTROPY
REDQ_TARGET_ENTROPY     = None   # None → default heuristic of -|action_size|
REDQ_BATCH_SIZE         = 512    # mini-batch for REDQ (bigger → better GPU utilisation)

# SAC parameters (Haarnoja et al. 2018 v2, auto-tuned α)
SAC_ALPHA_LR             = 3e-4   # optimiser lr for the temperature scalar
SAC_INIT_LOG_ALPHA       = 0.0    # log α(0) → α(0) = 1.0; tuned online
SAC_TARGET_ENTROPY_SCALE = 0.5    # target entropy = -|A| * scale (0.5 → less forced exploration, exploit learned policy harder; sac_5 value)
SAC_LOG_STD_MIN          = -20    # clamp on the Gaussian policy log_std for stability
SAC_LOG_STD_MAX          = 2

# Eval-based checkpoint selection: every MODEL_STORE_INTERVAL training eps,
# pause and run N deterministic-policy episodes; save `_best.pt` only when
# this score improves. Set to 0 to disable.
VAL_EPS_PER_CHECKPOINT   = 50

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