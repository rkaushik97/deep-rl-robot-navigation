# SAC experiment: reward VH = reward V + gated heading + control smoothness (exp007).
# Goal: push past reward V's 84% by fixing V's REMAINING wall failures (mid-journey
# "clip-while-navigating") without re-introducing the over-caution/timeout collapse.
#   + r_heading  = -0.10 * |goal_angle|/pi   ONLY when obs_dist >= OBSTACLE_SAFE (gated to open
#                  space, so it never reinforces driving straight into a wall)
#   + r_smooth   = -0.10 * (d_angular + 0.25*d_linear)  (damps jerky control / reversing)
# Same base as reward V (potential progress + speed*proximity obstacle penalty, O(1), no -2000 cliff).
# Sourced AFTER config.sh.
export DRL_REWARD=VH
export DRL_REWARD_SCALE=1.0
export DRL_PROGRESS_K=2.0                  # progress-shaping gain (same as reward P/V)
export DRL_OBSTACLE_K=0.5                  # peak speed*proximity obstacle penalty
export DRL_OBSTACLE_SAFE=0.40              # danger-zone distance (m); also gates the heading term
export DRL_SAC_TARGET_ENTROPY_SCALE=1.0    # target entropy -2.0 (keep exploring)
export DRL_TAU=0.005
