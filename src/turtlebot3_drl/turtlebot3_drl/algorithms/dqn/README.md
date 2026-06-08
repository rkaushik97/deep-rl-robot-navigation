# DQN

Reference-exact DQN for stage 9.

- `dqn.py` — Q-network + the DQN update (discrete action space, target network updates, epsilon-greedy exploration).
- `dqn_config.sh` — frozen reference hyperparameters. The BASE recipe — edit to change the base.
- `experiments/` — variations to try beating the baseline. Each file exports only the knobs
  it changes and is sourced AFTER `dqn_config.sh`.

## Train
```
scripts/train.sh dqn               # base reference config
scripts/train.sh dqn curriculum    # base + experiments/curriculum.sh
```

## Try something new
Add experiments/<name>.sh exporting just the knobs you change (e.g. export DRL_EPSILON_DECAY=500000),
then scripts/train.sh dqn <name>. Score it with scripts/eval.sh dqn <model_dir> <ep>
against the baseline.
