# DDPG — frozen reference-exact config (tomasvr/turtlebot3_drlnav, stage 9).
# Sourced by scripts/train.sh. Edit here to change the BASE recipe; for one-off
# variations add a file under experiments/ instead of editing this.
export DRL_REWARD=A                 # dense reward "A"
export DRL_REWARD_SCALE=1.0         # DDPG/TD3 magnitude
export DRL_DYNAMIC_GOALS=False      # no curriculum (random goals, like the eval)
export DRL_BACKWARD=True            # action space [-0.22, 0.22] (reverse allowed)
export DRL_BATCH_SIZE=256
export DRL_LR=0.0003
export DRL_TAU=0.003
export DRL_STEP_TIME=0.05           # ~20 Hz control cadence
# OU exploration noise (constant sigma)
export DRL_SIGMA_MAX=0.1
export DRL_SIGMA_MIN=0.1
export DRL_DECAY=8000000
# in-loop validation every 100 eps (feeds the val-success plot panel)
export DRL_VAL_EPS=40
