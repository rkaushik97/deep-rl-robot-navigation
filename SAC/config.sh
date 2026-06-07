# SAC — frozen reference-exact config (sac_5, stage 9).
# Sourced by scripts/train.sh. Edit here for the BASE recipe; use experiments/ for variations.
export DRL_REWARD=A
export DRL_REWARD_SCALE=0.1         # SAC entropy/Q balance (NOT 1.0)
export DRL_DYNAMIC_GOALS=False
export DRL_BACKWARD=True
export DRL_BATCH_SIZE=256
export DRL_LR=0.0003
export DRL_TAU=0.003
export DRL_STEP_TIME=0.05
# SAC-specific: auto-tuned temperature + squashed-Gaussian policy
export DRL_SAC_ALPHA_LR=3e-4
export DRL_SAC_INIT_LOG_ALPHA=0.0
export DRL_SAC_TARGET_ENTROPY_SCALE=0.5
export DRL_SAC_LOG_STD_MIN=-20
export DRL_SAC_LOG_STD_MAX=2
export DRL_VAL_EPS=40
