# exp002 — failure analysis (what exactly fails)

DDPG + sparse + potential-based progress, 831 eps, ~35% deterministic. Dominant failure: wall collisions.

## Outcome-level stats (whole run)
| outcome | n | mean steps | mean reward |
|---|---|---|---|
| SUCCESS | 207 | 159 | +3.9 |
| COLL_WALL | 590 | 152 | +0.2 |
| TIMEOUT | 25 | 814 | +0.6 |
| TUMBLE | 9 | 39 | +0.3 |

**Key fact: wall collisions last ~as long as successes (152 vs 159 steps).** The agent is NOT failing to move — it navigates, then clips a wall.

## Wall-collision episode-length distribution
| length (steps) | share | interpretation |
|---|---|---|
| <30 | 8% | instant — straight into wall (orientation/spawn) |
| 31–80 | 28% | quick |
| 81–160 | 30% | navigated a bit, then clipped |
| 161–300 | 24% | navigated far, then clipped |
| 300+ | 11% | long episode |

→ **~65% of crashes occur after 80+ steps of navigation.**

## Behavior at the collision moment (env debug)
At/near collisions: linear action 0.7–1.0 (near max forward) while min-obstacle-distance drops to 0.00–0.13 m (collision threshold = 0.13). **The agent does not decelerate or steer clear near obstacles** — it drives at near-max speed right into walls.
- Most collisions: heading roughly toward goal (small goal-angle) → clipped a wall en route.
- A minority: goal-angle 90–160° (goal behind/beside) → drove the wrong way into a wall (heading error).

## Diagnosis
Dominant failure mode: **"navigates toward the goal at near-max speed but clips walls — no obstacle-avoidance, no slowdown near obstacles."** The progress reward taught goal-seeking; nothing taught wall clearance.

→ Validates **exp003 = + obstacle-proximity penalty** (teach clearance + implicit slowdown near walls). Over-caution is not yet a risk (only 3% timeouts), so there is headroom.

(Per-collision cause breakdown quantified separately via deterministic replay — see replay_analysis.)

## Replay cause breakdown (best checkpoint, 40 greedy episodes)
40 episodes: 12 success (30%), 27 wall collisions (68%), 1 timeout.

Of the 27 wall collisions:
| cause | count | % of walls |
|---|---|---|
| **CLIP-WHILE-NAVIGATING (the problem)** | **20** | **74%** |
| wrong-heading (goal >100° off-axis) | 4 | 15% |
| early/orientation (<40 steps) | 3 | 11% |

- clip-while-navigating: 18 mid-journey, 2 near-goal (<0.5 m).
- **56% of all wall collisions happened at high forward speed (lin > 0.5)** — drives fast into walls, no slowdown.
- ≈ **74% of walls × 68% wall-rate ⇒ ~50% of ALL episodes fail via this one mode.**

→ Confirms exp003 (obstacle penalty) targets the right thing; the high-speed component argues the penalty should also induce slowdown near obstacles (the smooth proximity term does).
