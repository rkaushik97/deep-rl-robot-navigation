# Replications — DDPG & TD3 (stage 9)

Faithful reproductions of the reference repo's DDPG and TD3, trained from scratch with the
reference config and scored on its benchmark (`test_agent`, 100 random-goal episodes).

| Algorithm | Our test | Reference |
|-----------|----------|-----------|
| DDPG (ep4000) | **89%** | 84% |
| TD3 (ep2700)  | **80%** | 74% |

![DDPG](ddpg/curve_vs_reference.png)
![TD3](td3/curve_vs_reference.png)

Each folder has the training curve (`_metrics.tsv`, `training.png`), the frozen `config.sh`, the
best actor weights, and the verified test result.

**Reproduce:** `scripts/train.sh ddpg` (or `td3`), then `scripts/eval.sh ddpg <model_dir> <ep> 100`.
