# SAC

Reference-exact SAC for stage 9 (baseline: **82%** on the central eval).

- `sac.py` — squashed-Gaussian actor + twin Critic, auto-tuned temperature (alpha).
- `config.sh` — frozen reference hyperparameters (note `DRL_REWARD_SCALE=0.1`, SAC entropy knobs). The BASE recipe.
- `experiments/` — variations to try beating the baseline (sourced AFTER `config.sh`).

## Train
```
scripts/train.sh sac            # base reference config
scripts/train.sh sac nstep3     # base + experiments/nstep3.sh (3-step returns)
```

## Try something new
Add `experiments/<name>.sh` exporting just the knobs you change, then
`scripts/train.sh sac <name>`. Score with `scripts/eval.sh sac <model_dir> <ep>` vs the 82% baseline.
