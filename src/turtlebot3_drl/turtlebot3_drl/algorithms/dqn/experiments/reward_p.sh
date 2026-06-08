# DQN reward-P experiment. Sourced AFTER config.sh; overrides only these knobs.
# Motivation: base reward A gives DQN exploding Q-loss (~1e4) and a ~28% plateau.
# Reward P (sparse +/-1 + potential-based progress) keeps Q ~O(10) -> stable learning.
export DRL_REWARD=P                # sparse +/-1 + progress (replaces dense reward A)
export DRL_EPSILON_DECAY=0.99995   # slower anneal (~hundreds of eps); 0.9995 floored in ~30-50 eps
export DRL_VAL_EPS=0               # skip in-loop 40-goal validation; eval with standard test_agent at the end
