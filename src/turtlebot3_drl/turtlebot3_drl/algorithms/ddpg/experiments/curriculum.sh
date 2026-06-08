# Curriculum learning: goal difficulty (ring radius) adapts to success rate.
# Sourced AFTER config.sh, so it only overrides what it names.
export DRL_DYNAMIC_GOALS=True
# Real easy->hard ramp INSIDE the 4.2m arena (valid coords |x|,|y|<=2.1, so from the
# reset origin the farthest reachable goal is ~2.97m). The old default [2.0,4.0] had no
# easy start (2.0m is already hard) and a 4.0m ceiling that can't be placed (silent
# fallback to fixed goals). 1.0 = genuine easy start; 2.8 stays placeable.
export DRL_CURRICULUM_MIN=1.0
export DRL_CURRICULUM_MAX=2.8
