# SAC experiment: sparse + potential-based progress reward, standard SAC entropy target.
# Rationale: reward A (dense, +-2500) is mis-scaled for SAC's entropy objective and its huge
# -2000 collision penalty drove a risk-averse spin-and-timeout local optimum (58% timeouts,
# policy degrading). Reward P is O(1) (sparse +-1 terminal + potential-based progress, which is
# policy-invariant and can't be reward-hacked), and target entropy -|A|=-2.0 keeps it exploring
# long enough to find goal-seeking before it commits. Sourced AFTER config.sh.
export DRL_REWARD=P                       # sparse +-1 terminal + potential-based progress (O(1))
export DRL_REWARD_SCALE=1.0               # P is already O(1) — no down-scaling
export DRL_PROGRESS_K=2.0                 # progress-shaping gain
export DRL_SAC_TARGET_ENTROPY_SCALE=1.0   # target entropy = -|A| = -2.0 (SAC default; more exploration)
export DRL_TAU=0.005                      # standard SAC soft-update (base config uses 0.003)
