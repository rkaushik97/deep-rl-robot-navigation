# DQN — reference config adapted for discrete action spaces
# Sourced by scripts/train.sh. Edit here to change the BASE recipe.

# --- Environment & Reward Settings ---
export DRL_REWARD=A                 # dense reward "A"
export DRL_DYNAMIC_GOALS=False      # no curriculum (random goals, like the eval)
export DRL_BACKWARD=True            # linear maps linearly: 0.3 -> 30% speed, 1.0 -> full speed
export DRL_STEP_TIME=0.05           # ~20 Hz control cadence
export DRL_VAL_EPS=40               # in-loop validation every 100 eps

# --- General Neural Network Settings ---
export DRL_BATCH_SIZE=128           # Standard batch size for DQN
export DRL_LR=0.0001                # Learning rate for the Q-network
export DRL_GAMMA=0.99               # Discount factor for future rewards

# --- DQN-Specific Settings ---
export DRL_ACTION_SIZE=5            # Number of discrete actions (e.g., forward, left, right, reverse, stop)
export DRL_TARGET_UPDATE_FREQ=1000  # Steps before copying weights to Target Q-Network

# --- Epsilon-Greedy Exploration ---
# Epsilon starts at 1.0 and is multiplied by DRL_EPSILON_DECAY once per train iteration
# until it reaches DRL_EPSILON_MIN.
export DRL_EPSILON_MIN=0.05         # Minimum exploration rate (exploit 95% of the time)
export DRL_EPSILON_DECAY=0.9995     # Per-iteration multiplicative decay