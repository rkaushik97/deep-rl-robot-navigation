import os
# ===================================================================== #
#                           GENERAL SETTINGS                            #
# ===================================================================== #
# Experiment-overridable knobs (set by scripts/run_experiment.sh per run, so each
# experiment is reproducible without editing/rebuilding). Defaults below apply when
# the env vars are unset.

ENABLE_BACKWARD          = (os.environ.get('DRL_BACKWARD', 'False') == 'True')   # env: DRL_BACKWARD. False = forward-only [0,0.22] (DDPG example). True = [-0.22,0.22] reverse allowed (the reference TD3 example used True).
# TD3 target-policy-smoothing / delayed-update params (Fujimoto et al. 2018) — match reference TD3 example.
POLICY_NOISE            = float(os.environ.get('DRL_POLICY_NOISE', '0.2'))
POLICY_NOISE_CLIP       = float(os.environ.get('DRL_POLICY_NOISE_CLIP', '0.5'))
POLICY_UPDATE_FREQUENCY = int(os.environ.get('DRL_POLICY_FREQ', '2'))
ENABLE_STACKING          = (os.environ.get('DRL_STACKING', '0') == '1')   # temporal obs: stack last STACK_DEPTH frames (env: DRL_STACKING). Lets the policy infer obstacle MOTION on dynamic stage 9. NOTE: changes input_size 44->132 -> warm-start INCOMPATIBLE (train from scratch).
ENABLE_VISUAL            = False    # Meant to be used only during evaluation/testing phase
ENABLE_TRUE_RANDOM_GOALS = False    # If false, goals are selected semi-randomly from a list of known valid goal positions
ENABLE_DYNAMIC_GOALS     = (os.environ.get('DRL_DYNAMIC_GOALS', 'True') == 'True')   # curriculum on/off (env: DRL_DYNAMIC_GOALS). Goal spawns on a difficulty_radius ring, clamped to [CURRICULUM_MIN_RADIUS, CURRICULUM_MAX_RADIUS].
MAX_TRAINING_EPISODES    = int(os.environ.get('DRL_MAX_EPISODES', '0'))              # stop training after N episodes (env: DRL_MAX_EPISODES); 0 = unlimited.
TEST_EPISODES            = int(os.environ.get('DRL_TEST_EPISODES', '0'))             # stop test_agent after N episodes then print summary (env: DRL_TEST_EPISODES); 0 = unlimited.
CURRICULUM_MIN_RADIUS    = float(os.environ.get('DRL_CURRICULUM_MIN', '2.0'))   # floor (env: DRL_CURRICULUM_MIN). 2.0 keeps goals non-trivial for FIXED-difficulty runs; lower (e.g. 1.0) enables a PROGRESSIVE curriculum that ramps up via hysteresis (exp014). Eval env uses default 2.0 -> fair benchmark.
CURRICULUM_MAX_RADIUS    = float(os.environ.get('DRL_CURRICULUM_MAX', '4.0'))   # cap (env: DRL_CURRICULUM_MAX); benchmark static goals reach ~3.4m, so [2.0,4.0] spans real-navigation difficulty.
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
REWARD_SCALE    = float(os.environ.get('DRL_REWARD_SCALE', '1.0'))   # env: DRL_REWARD_SCALE. 1.0 = DDPG/TD3 magnitude; 0.1 for SAC's entropy/Q balance. Scales reward MAGNITUDE only.
# SAC parameters (Haarnoja et al. 2018 v2, auto-tuned temperature alpha) — match reference SAC (kaushik/sac, sac_5).
SAC_ALPHA_LR             = float(os.environ.get('DRL_SAC_ALPHA_LR', '3e-4'))
SAC_INIT_LOG_ALPHA       = float(os.environ.get('DRL_SAC_INIT_LOG_ALPHA', '0.0'))
SAC_TARGET_ENTROPY_SCALE = float(os.environ.get('DRL_SAC_TARGET_ENTROPY_SCALE', '0.5'))
SAC_LOG_STD_MIN          = int(os.environ.get('DRL_SAC_LOG_STD_MIN', '-20'))
SAC_LOG_STD_MAX          = int(os.environ.get('DRL_SAC_LOG_STD_MAX', '2'))
PROGRESS_K      = float(os.environ.get('DRL_PROGRESS_K', '2.0'))     # reward_P/_O: gain on closing-distance shaping (potential-based, policy-invariant). env: DRL_PROGRESS_K.
OBSTACLE_K      = float(os.environ.get('DRL_OBSTACLE_K', '0.5'))     # reward_O: peak obstacle-proximity penalty at contact (quadratic ramp). env: DRL_OBSTACLE_K.
OBSTACLE_SAFE   = float(os.environ.get('DRL_OBSTACLE_SAFE', '0.40')) # reward_O: distance (m) within which the quadratic obstacle penalty ramps from 0 (at SAFE) to -OBSTACLE_K (contact). env: DRL_OBSTACLE_SAFE.
# OU exploration-noise schedule (anneals max->min over decay_period GLOBAL steps). Default = constant 0.1 (legacy tomasvr value). env: DRL_SIGMA_MAX/DRL_SIGMA_MIN/DRL_DECAY.
NOISE_SIGMA_MAX    = float(os.environ.get('DRL_SIGMA_MAX', '0.1'))
NOISE_SIGMA_MIN    = float(os.environ.get('DRL_SIGMA_MIN', '0.1'))
NOISE_DECAY_PERIOD = int(float(os.environ.get('DRL_DECAY', '8000000')))
ACTION_SIZE     = 2         # continuous algos: [linear, angular]. DQN uses DQN_ACTION_SIZE.
DQN_ACTION_SIZE = int(os.environ.get('DRL_ACTION_SIZE', '5'))   # DQN: number of discrete actions
HIDDEN_SIZE     = 512       # 1024 was tried for SAC v2 — caused policy collapse, reverted

BATCH_SIZE      = int(os.environ.get('DRL_BATCH_SIZE','128'))  # Reference winning-DDPG value (tomasvr/turtlebot3_drlnav). 1024 + low lr starved effective updates -> policy collapse on stage 9.
BUFFER_SIZE     = 1000000   # Number of samples stored in replay buffer before FIFO
DISCOUNT_FACTOR = float(os.environ.get('DRL_GAMMA', '0.99'))
# N-step returns (env: DRL_NSTEP). 1 = exact 1-step TD (default, byte-identical). 3..5 speeds
# credit assignment on the sparse/dynamic task. No network-shape change -> warm-start/resume safe.
N_STEP = int(os.environ.get('DRL_NSTEP', '1'))
LEARNING_RATE   = float(os.environ.get('DRL_LR','0.003'))  # Reference winning-DDPG value. 3e-4 was 10x too slow: actor locked in the r_vlinear max-speed habit before the critic learned obstacle avoidance -> collapse.
TAU             = float(os.environ.get('DRL_TAU','0.003'))  # Reference winning-DDPG value (NOT a "SAC value"). 3e-4 made target tracking 10x too slow.

OBSERVE_STEPS   = 25000     # At training start random actions are taken for N steps for better exploration
STEP_TIME       = float(os.environ.get('DRL_STEP_TIME','0.01'))  # Reference winning-DDPG value: a defined, consistent action-execution window. 0.0 made the MDP timestep equal to variable ROS service latency (ill-defined).
EPSILON_DECAY   = float(os.environ.get('DRL_EPSILON_DECAY', '0.9995'))  # DQN: per-train-iter multiplicative decay
EPSILON_MINIMUM = float(os.environ.get('DRL_EPSILON_MIN', '0.05'))
TARGET_UPDATE_FREQUENCY = int(os.environ.get('DRL_TARGET_UPDATE_FREQ', '1000'))  # DQN: hard target sync every N train iters

# Eval-based checkpoint selection: every MODEL_STORE_INTERVAL training eps,
# pause and run N deterministic-policy episodes; save `_best.pt` only when
# this score improves. Set to 0 to disable.
VAL_EPS_PER_CHECKPOINT   = int(os.environ.get('DRL_VAL_EPS', '0'))   # in-loop n=20 deterministic validation: DISABLED by default (env: DRL_VAL_EPS). It was noisy/useless (n=20 swings +-20pp on a slowly-improving policy) AND wasted training time pausing every 100 eps AND its *_best.pt selection was unreliable (stale-ckpt bugs). Periodic checkpoints (every MODEL_STORE_INTERVAL) still save -> pick the best via the 40-ep clean replay (scripts/replay_analyze.py) instead.

# Stacking (env: DRL_STACK_DEPTH / DRL_FRAME_SKIP). Agent keeps last (DEPTH-1)*SKIP+1 raw
# frames and emits frames at offsets [0, -SKIP, -2*SKIP, ...] (newest first).
STACK_DEPTH = int(os.environ.get('DRL_STACK_DEPTH', '3'))   # frames in the stacked observation
FRAME_SKIP  = int(os.environ.get('DRL_FRAME_SKIP', '4'))    # step gap between stacked frames

# Episode outcome enumeration
UNKNOWN = 0
SUCCESS = 1
COLLISION_WALL = 2
COLLISION_OBSTACLE = 3
TIMEOUT = 4
TUMBLE = 5
RESULTS_NUM = 6