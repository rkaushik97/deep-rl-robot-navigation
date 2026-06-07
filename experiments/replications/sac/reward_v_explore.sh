# SAC experiment: reward V = reward P + speed-modulated obstacle penalty.
# Targets the ~25% wall floor (fast "clip-while-navigating") that reward P plateaus on (~73% test),
# WITHOUT re-introducing the timeout trap: the penalty is gated on speed*proximity, so it is ~0
# when slow (even near a wall) or far (even at full speed), and maximal only when FAST AND CLOSE.
#   r_speed = -OBSTACLE_K * (forward_speed/MAX) * ((SAFE - obs_dist)/SAFE)   [only if obs_dist < SAFE]
# Keeps reward P's good properties (O(1), no -2000 cliff, potential-based progress). Sourced AFTER
# config.sh. If timeouts creep back, OBSTACLE_K is too high (over-penalizing) — tune it down.
export DRL_REWARD=V
export DRL_REWARD_SCALE=1.0
export DRL_PROGRESS_K=2.0                  # progress-shaping gain (same as reward P)
export DRL_OBSTACLE_K=0.5                  # peak obstacle penalty (fast + at-contact)
export DRL_OBSTACLE_SAFE=0.40              # danger-zone distance (m)
export DRL_SAC_TARGET_ENTROPY_SCALE=1.0    # target entropy -2.0 (keep exploring)
export DRL_TAU=0.005
