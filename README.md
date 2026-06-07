# deep-rl-robot-navigation

Deep RL for TurtleBot3 goal-navigation among 6 moving obstacles (ROS 2 + Gazebo, stage 9).
DDPG and TD3 reproduce the reference implementation; SAC is a new from-scratch contribution.

## Results (test_agent — 100 random-goal episodes)

| Algorithm | Test | Reference | |
|-----------|------|-----------|--|
| DDPG | **89%** | 84% | replication — [experiments/replications/](experiments/replications) |
| TD3  | **80%** | 74% | replication |
| SAC  | **84%** | 82% | new contribution — [SAC/](SAC) |

## Layout
```
SAC/                    the SAC contribution (results, configs, best weights)
experiments/
  replications/         DDPG & TD3 results
  reference_checkpoints/ reference DDPG/TD3/SAC checkpoints
src/turtlebot3_drl/     the ROS 2 package
  turtlebot3_drl/algorithms/   ddpg/ td3/ sac/ + base.py
  turtlebot3_drl/training/     live display, _metrics.tsv, training.png
  turtlebot3_drl/evaluation/   the test_agent benchmark
  turtlebot3_drl/{common,drl_environment,drl_gazebo}/
scripts/                train.sh, eval.sh
```

## Install
ROS 2 Jazzy + Gazebo Harmonic (Ubuntu 24.04) + PyTorch.
```bash
rosdep install --from-paths src --ignore-src -r -y
pip install torch numpy matplotlib pyyaml
colcon build --symlink-install && source install/setup.bash
```

## Train & evaluate
```bash
scripts/train.sh ddpg                          # train (also: td3, sac)
scripts/eval.sh  ddpg <model_dir> <episode>    # test_agent, 100 episodes
```
Training prints a live line and writes `_metrics.tsv` + `training.png` to its session dir.

## Credits
DDPG, TD3, and the robot/environment settings are adapted from
[prakash-aryan/turtlebot3_deepRL](https://github.com/prakash-aryan/turtlebot3_deepRL).
