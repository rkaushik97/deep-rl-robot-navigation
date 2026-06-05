# TD3

Reference-exact TD3 for stage 9 (baseline: **74%** on the central eval — timeout-heavy).

- `td3.py` — Actor + twin Critic, target-policy smoothing, delayed actor updates.
- `config.sh` — frozen reference hyperparameters (adds `DRL_POLICY_NOISE/CLIP/FREQ`). The BASE recipe.
- `experiments/` — variations to try beating the baseline (sourced AFTER `config.sh`).

## Train
```
scripts/train.sh td3            # base reference config
scripts/train.sh td3 per        # base + experiments/per.sh (prioritized replay)
```

## Try something new
Add `experiments/<name>.sh` exporting just the knobs you change, then
`scripts/train.sh td3 <name>`. Score with `scripts/eval.sh td3 <model_dir> <ep>` vs the 74% baseline.
