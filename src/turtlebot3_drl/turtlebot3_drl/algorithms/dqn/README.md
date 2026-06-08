# DQN

Discrete-action DQN for stage 9. The policy picks one of 5 discrete `[linear, angular]`
actions (see `POSSIBLE_ACTIONS` in `dqn.py`); the harness maps the chosen index to velocities.

- `dqn.py` — Q-network + the DQN update (discrete actions, hard target sync, epsilon-greedy).
- `config.sh` — base hyperparameters. The BASE recipe — edit to change the base.
- `experiments/` — variations to try beating the baseline. Each file exports only the knobs
  it changes and is sourced AFTER `config.sh`.

## Train
```
scripts/train.sh dqn               # base config
scripts/train.sh dqn curriculum    # base + experiments/curriculum.sh
```

## Try something new
Add experiments/<name>.sh exporting just the knobs you change (e.g. export DRL_EPSILON_DECAY=0.999),
then scripts/train.sh dqn <name>. Score it with scripts/eval.sh dqn <model_dir> <ep>
against the baseline.
