# Replications — DDPG & TD3 (stage 9)

Faithful replications of the reference repo's DDPG and TD3, trained from scratch here with the
exact reference config and scored on the **same benchmark** (`test_agent`, 100 random-goal
deterministic episodes). SAC is still training and is not included here yet.

## Result (test_agent, 100 episodes)

| Algorithm | Our checkpoint | **Our test** | Reference test | At episode |
|-----------|----------------|--------------|----------------|------------|
| DDPG | ep4000 | **89%** | 84% | 4000 (ref: 8000) |
| TD3  | ep2700 | **80%** | 74% | 2700 (ref: 7400) |

Both **match and modestly beat** the reference (+5pp / +6pp; ~1–1.3σ at n=100), and reach it at
roughly half the reference's episode count. Numbers are sampling-noisy (±~3.5–4.6pp at n=100).

## What's here
Per algorithm (`ddpg/`, `td3/`):
- `_metrics.tsv` — full per-episode training log (MA100 success, losses, reward components).
- `training.png` — 4-panel training figure (success MA100, val success, losses, reward components).
- `config.sh` — the frozen reference-exact config used to train.
- `actor_stage9_episode<E>_best.pt` — the evaluated best actor weights.
- `test_agent_ep<E>_100eps.txt` — the verified benchmark result.
- `training_vs_reference.png` — our training curves vs the reference's, at matched episodes.

## Reproduce
```
scripts/train.sh ddpg          # or td3
scripts/eval.sh  ddpg <model_dir> <episode> 100    # test_agent, 100 random-goal eps
```
See `src/turtlebot3_drl/turtlebot3_drl/evaluation/README.md` for the benchmark definition.
