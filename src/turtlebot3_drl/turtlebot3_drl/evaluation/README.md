# Evaluation — the one benchmark everything is scored against

This is the **central test set**: the reference repo's `test_agent` methodology, run
natively here. Every algorithm — the legacy DDPG/TD3/SAC and anything new — is scored
exactly the same way, so numbers are comparable.

## What one test episode is
- The robot spawns and gets a **random goal** (`ENABLE_DYNAMIC_GOALS=False` → goals are
  drawn from the full distribution, no curriculum, no fixed list).
- The 6 animated obstacles keep moving; the obstacle phase is **not** reset between episodes.
- The policy runs **deterministically** (no exploration noise; SAC uses `tanh(mu)`).
- The episode ends as **SUCCESS** (reached goal, <0.20 m), **COLLISION_WALL**,
  **COLLISION_OBSTACLE**, or **TIMEOUT** (50 sim-seconds).

Because goals are random and unseeded, two runs of the same checkpoint differ by a few
points — report the rate over a decent N (we use 100). This is the same eval that produced
the reference's published numbers.

## Run it
```
scripts/eval.sh <algo> <model_dir> <episode> [N=100] [domain=54]
```
`<model_dir>` is a session folder containing `actor_stage9_episode<EP>.pt`. It is symlinked
under `model/examples/` so it loads regardless of which machine trained it. `test_agent`
stops itself after N episodes (`DRL_TEST_EPISODES`) and prints the summary.

## Output
A one-screen summary plus a copy in `results/`:
```
episodes = 100  SUCCESS = 84 (84%)  wall = 14 (14%)  obstacle = 0 (0%)  timeout = 2 (2%)
```

## Read a raw test log directly
```
python3 -m turtlebot3_drl.evaluation.evaluate <_test_stage9_eps*.txt> "label"
```
The log's last column is cumulative `success/wall/obstacle/timeout/tumble`.

## Reference baselines (measured by this harness)
| algorithm | checkpoint | success |
|-----------|-----------|---------|
| DDPG | ep8000 | 84% |
| TD3  | ep7400 | 74% |
| SAC  | ep6400 | 82% |
