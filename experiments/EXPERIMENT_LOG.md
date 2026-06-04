# Experiment Log — DDPG TurtleBot3 stage-9 navigation

**Goal:** 95% training / ≥90% deterministic-eval success, **DDPG only**.
**Autonomous run** (operator away ~8h). Decision rules:
- Metric = clean deterministic eval (40-ep greedy replay on best ckpt). Beat to chase: **exp002 = 40%**.
- Commit + push to `main` (safe helper, no binaries) on any **≥5pp** clean-eval improvement over last pushed best.
- One variable per experiment; warm-start from best policy so far; judge by clean eval + 100-ep training MA (never n=20 point evals).
- Stay DDPG (TD3/SAC removed per operator). Levers: reward shaping, exploration-noise decay, hyperparameters, curriculum, training length.

## Methodology
Sparse-first (unhackable) → add one dense component at a time → verify by clean eval. Per-experiment folders, comparison plot `_comparison_success_ma.png` (100-ep MA, robust peaks).

## Results so far
| exp | change | clean eval | notes |
|---|---|---|---|
| exp001 | DDPG + sparse (+1/-1) | 0% | healthy losses, exploration-limited (never reaches goal) |
| exp002 | + potential-based progress | **40%** | broke the 0% wall; PUSHED baseline. dominant failure: clip-while-navigating (~85% of walls) |
| exp003 | + obstacle-proximity penalty (approach-aware) | 45% | within noise of exp002 → NEUTRAL. (static proximity v1 collapsed; approach-aware fixed collapse but no gain) |
| exp004 | + speed-modulated penalty (slow-in-clutter) | running | targets the 56% high-speed clips; warm-start exp002 |

## Failure analysis (exp002/exp003, 40-ep replay)
- ~85% of wall collisions = "clip while navigating" (drives toward goal, clips a wall en route); 56% at high forward speed.
- wrong-heading ~0% in clean eval (earlier 15% was training noise). early/orientation ~12-18% of walls.
- Lever that matters: stop high-speed clips (→ speed-modulated penalty), and close the train/eval gap (→ exploration-noise decay).

## Planned pipeline (will adapt to results)
1. exp004 — speed-modulated penalty (RUNNING)
2. exp005 — exploration-noise decay (anneal OU sigma) — closes the train/eval gap, stabilises late training
3. exp006 — tune best reward (PROGRESS_K / OBSTACLE_K / SAFE) around the winner
4. exp007 — hyperparameters (lr, batch) on the winner
5. exp008 — longer training of the best + best-checkpoint capture
6. exp009 — curriculum tuning (ramp difficulty toward full-distance goals)

## Running log (chronological)
- exp004 launched (warm-start exp002 best + reward V speed-modulated). Monitoring.
