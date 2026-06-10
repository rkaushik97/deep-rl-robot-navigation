# SAC reward V + curriculum learning. Reward V is the SAC winner (84%); this tests whether
# the adaptive goal-radius curriculum (ring 1.0->2.8 m, ×1.02/×0.99 on success/fail) speeds or
# improves it. Sourced AFTER config.sh, so it overrides only these knobs.
# --- reward V (same as reward_v_explore.sh) ---
export DRL_REWARD=V
export DRL_REWARD_SCALE=1.0
export DRL_PROGRESS_K=2.0
export DRL_OBSTACLE_K=0.5
export DRL_OBSTACLE_SAFE=0.40
export DRL_SAC_TARGET_ENTROPY_SCALE=1.0
export DRL_TAU=0.005
# --- curriculum (same fixed, arena-valid bounds as the DDPG curriculum run) ---
export DRL_DYNAMIC_GOALS=True
export DRL_CURRICULUM_MIN=1.0
export DRL_CURRICULUM_MAX=2.8
