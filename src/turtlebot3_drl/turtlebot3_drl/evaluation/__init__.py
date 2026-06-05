"""Central evaluation — the reference repo's `test_agent` methodology, native here.

Every algorithm is scored the SAME way: run the deterministic policy on N episodes
with random goals (ENABLE_DYNAMIC_GOALS=False), the obstacle phase running freely,
and tally SUCCESS / COLLISION_WALL / COLLISION_OBSTACLE / TIMEOUT. This is the one
benchmark the whole repo compares against. See README.md in this folder.
"""
