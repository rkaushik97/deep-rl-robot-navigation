# TD3 — frozen reference-exact config (stage 9).
# Sourced by scripts/train.sh. Edit here for the BASE recipe; use experiments/ for variations.
export DRL_REWARD=A
export DRL_REWARD_SCALE=1.0
export DRL_DYNAMIC_GOALS=False
export DRL_BACKWARD=True
export DRL_BATCH_SIZE=256
export DRL_LR=0.0003
export DRL_TAU=0.003
export DRL_STEP_TIME=0.05
# TD3-specific: target-policy smoothing + delayed actor updates
export DRL_POLICY_NOISE=0.2
export DRL_POLICY_NOISE_CLIP=0.5
export DRL_POLICY_FREQ=2
# OU exploration noise (constant sigma)
export DRL_SIGMA_MAX=0.1
export DRL_SIGMA_MIN=0.1
export DRL_DECAY=8000000
export DRL_VAL_EPS=40
