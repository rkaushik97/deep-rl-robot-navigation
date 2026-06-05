# DDPG

Reference-exact DDPG for stage 9 (baseline: **84%** on the central eval).

- `ddpg.py` — Actor/Critic networks + the DDPG update (deterministic policy gradient,
  twin target networks, OU exploration noise).
- `config.sh` — frozen reference hyperparameters. The BASE recipe — edit to change the base.
- `experiments/` — variations to try beating the baseline. Each file exports only the knobs
  it changes and is sourced AFTER `config.sh`.

## Train
```
scripts/train.sh ddpg               # base reference config
scripts/train.sh ddpg curriculum    # base + experiments/curriculum.sh
```

## Try something new
Add `experiments/<name>.sh` exporting just the knobs you change (e.g. `export DRL_PER=1`),
then `scripts/train.sh ddpg <name>`. Score it with `scripts/eval.sh ddpg <model_dir> <ep>`
against the 84% baseline.
