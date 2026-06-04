# DDPG on stage 9 — final result (closed)

**Verdict: DDPG solves stage-9 navigation but is capped ~60% and unstable. Moving to TD3.**

## Final numbers (single-agent, winning-repo config: reward A, static goals, lr/tau 3e-3, batch 128, Huber+AdamW, OU σ=0.1)
- **Deterministic eval (no noise), peak checkpoint ep700: 12/20 = 60%** (ep900: 10/20 = 50%).
- Noisy training-success peak: ~55% (100-ep moving avg) around episode ~700.
- After the peak the policy **degrades** to ~33% by ep ~1300 — classic DDPG instability (single-critic overestimation; `−Q` crept 273 → 306 while success fell).
- Best policy preserved here: `runs/ddpg_final_best_ep700/` (actor+critic+targets) and curve `ddpg_11_single_success_ma100.png`.

## Why it's capped / unstable
Vanilla DDPG = single critic + deterministic actor → overestimation bias (the deadly triad). It learns, peaks, then the actor exploits an increasingly overestimated Q and drifts off the peak. No twin critic to bound it, no target-policy smoothing, no best-checkpoint capture on the original run.

## Distributed DDPG
Collapses entirely (~2% success, `−Q` diverges) under the custom async actor-learner pipeline — confirmed all-else-equal and at N=1. Root cause is the distributed act-learn architecture (not algo/reward/HP/staleness/multi-actor/sim-timing — all ruled out). Closed in favor of TD3 + a proven synchronous/vectorized parallelization later. See memory `ddpg-hyperparams-inverted-vs-reference`.

## What carried forward to TD3
- Best-checkpoint selection enabled (`VAL_EPS_PER_CHECKPOINT=20`) → captures the peak, logs `_eval_stage9.tsv`.
- Same task (reward A, static goals) for a clean DDPG-vs-TD3 comparison.
- Deterministic eval harness: `drl-distributed/scripts/{run_eval.sh,eval_checkpoint.py}`.
