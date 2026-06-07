# SAC (stage 9) — diagnosis, fix, and result

Unlike DDPG/TD3, SAC did **not** replicate on the reference's reward (reward A). This folder
documents the failure, the fix, and where it landed.

## Result (test_agent, 100 random-goal eps)

| Reward | Best test | Profile | vs reference |
|--------|-----------|---------|--------------|
| A (reference dense) | ~55–65%, **collapsed** | 58% timeout, policy degrading | — |
| P (sparse + progress) | 77% (ep1900) | 0–1% timeout, ~25% wall floor | 82% |
| **V (P + obstacle penalty)** | **83%** (ep3800) | 0% timeout, **17% wall** | **matches 82%** |

Reward P broke the collapse and reached a ~73–77% plateau but hit a ~25% wall floor. **Reward V**
added a speed-modulated obstacle penalty that broke that floor (walls 25% → 17%) and reached
**83% — matching the reference's 82%.** Note the SAC policy is noisy between checkpoints: the same
reward-V run tests **66% at ep3400 but 83% at ep3800**, so the val-selected `best.pt` matters
(here the val correctly flagged ep3800).

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
healthy ~73–77% test plateau, capped by a ~25% wall floor (reward P has no clearance term).

## Closing the wall floor — reward V
- **Reward V** (`get_reward_V`): reward P **plus** a speed-modulated obstacle penalty
  `r_speed = −OBSTACLE_K · (forward_speed/MAX) · ((SAFE − obs_dist)/SAFE)`, active only inside
  `OBSTACLE_SAFE=0.40 m`, with `OBSTACLE_K=0.5`. It penalizes being *fast AND close* (≈0 when slow
  or far), so it teaches "slow down in clutter" without forbidding being near walls — avoiding the
  over-caution timeout trap a blanket proximity penalty (reward O) would cause. See
  `reward_v_explore.sh`.
- **Result: walls 25% → 17%, test 83% @ ep3800 — matching the reference's 82%.** This is the best
  SAC result. (Watch the checkpoint noise: an arbitrary recent checkpoint can test ~66%; use the
  val-selected best.)

## Contents
`_metrics.tsv`, `training.png`, `curve_vs_reference.png`, `config.sh`, `reward_p_explore.sh` +
`reward_v_explore.sh`, `actor_stage9_episode1900_best.pt` (reward P best, 77%) +
`actor_stage9_episode3800_rewardV_best.pt` (reward V best, 83%), `train.log`, `test_agent_evals.txt`.

## Reproduce
```
scripts/train.sh sac reward_p_explore     # the collapse fix (~73-77%)
scripts/train.sh sac reward_v_explore     # + obstacle penalty (~83%, best)
scripts/eval.sh  sac <model_dir> <episode> 100
```
