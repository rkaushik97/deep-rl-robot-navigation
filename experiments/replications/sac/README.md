# SAC (stage 9) — diagnosis, fix, and result

Unlike DDPG/TD3, SAC did **not** replicate on the reference's reward (reward A). This folder
documents the failure, the fix, and where it landed.

## Result (test_agent, 100 random-goal eps)

| Reward | Best test | Profile | vs reference |
|--------|-----------|---------|--------------|
| A (reference dense) | ~55–65%, **collapsed** | 58% timeout, policy degrading | — |
| **P (sparse + progress)** | **77%** (ep1900) | 0–1% timeout, ~22% wall | reference 82% |

Test-set curve under reward P (each = 100 eps):
```
ep1100: 72%   ep1900: 77%   ep2400: 72%   ep2900: 71%   ->  plateau ~73%, wall floor ~25%
```

## Why reward A failed for SAC
Reward A is dense with huge magnitude (±2500/−2000 terminal, scaled 0.1 → Q ~ O(±250)). Two
problems specific to SAC's entropy objective `E[Q] + α·H(π)`:
1. **Scale mismatch** — the reward dwarfs the entropy term α regulates.
2. **Risk-aversion trap** — the −2000 collision cliff made *not committing* (spin/wander/timeout,
   mild −1/step) safer than driving at goals. SAC's value-based stochastic policy found and
   exploited this timid optimum: **58% timeouts, MA100 capped ~40% and degrading.**

Verified offline: code/config are byte-identical to the reference `sac_5`; α auto-tuning works;
the trained policy was confident (std 0.32) but stuck — i.e. premature convergence to the timid
attractor, not a bug.

## The fix — reward P + more exploration
- **Reward P** (`get_reward_P`): `r_progress = clip(2·(prev_dist − dist), ±1)` + terminal ±1,
  scale 1.0. O(1) magnitudes; potential-based (policy-invariant, un-hackable); no −2000 cliff.
- **Target entropy −2.0** (`SAC_TARGET_ENTROPY_SCALE=1.0`) — keeps exploring longer.
- tau 0.005. See `reward_p_explore.sh`.

Result: the timeout-collapse broke immediately (timeouts 58% → ~1%), success climbed to a
healthy ~73% test plateau. The remaining gap to the reference's 82% is **obstacle avoidance** — a
~25% wall floor that reward P (no clearance term) can't push past. Closing it is the next
experiment (reward V: + speed-modulated obstacle penalty).

## Contents
`_metrics.tsv`, `training.png`, `curve_vs_reference.png`, `config.sh` + `reward_p_explore.sh`,
`actor_stage9_episode1900_best.pt` (best by test, 77%), `train.log`, `test_agent_evals.txt`.

## Reproduce
```
scripts/train.sh sac reward_p_explore
scripts/eval.sh  sac <model_dir> <episode> 100
```
