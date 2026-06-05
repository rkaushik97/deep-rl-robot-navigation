# deep-rl-robot-navigation

Deep RL for TurtleBot3 goal-navigation among moving obstacles (ROS 2 + Gazebo, stage 9 =
6 animated obstacles). The repo replicates DDPG, TD3, and SAC from the reference
implementation, scores every algorithm on one shared benchmark, and gives each algorithm a
clean folder to try improvements against that baseline.

## Layout
```
DDPG/ TD3/ SAC/   per-algorithm code, frozen reference config, and an experiments/ playground
                  (symlinks to src/turtlebot3_drl/turtlebot3_drl/algorithms/<algo>/)
eval/             the ONE benchmark every algorithm is scored on (test_agent, random goals)
src/turtlebot3_drl/turtlebot3_drl/
  algorithms/     ddpg/ td3/ sac/ + base.py + REGISTRY (single dispatch point)
  evaluation/     the central eval (evaluate.py, README, results/)
  training/       universal harness: live display, _metrics.tsv, training.png
  common/         settings, replay buffer, storage, utilities, OU noise
  drl_environment/ drl_gazebo/   the simulator backend (ROS nodes)
scripts/          train.sh, eval.sh, plot.sh
```

## Install
Requires **ROS 2 Jazzy** and **Gazebo Harmonic** (gz-sim 8) on Ubuntu 24.04, plus PyTorch.
```bash
# ROS 2 Jazzy + Gazebo Harmonic must already be installed (ros-jazzy-desktop, ros-jazzy-ros-gz).
git clone <this repo> && cd deep-rl-robot-navigation
rosdep install --from-paths src --ignore-src -r -y      # ROS deps
pip install torch numpy matplotlib pyyaml                # Python deps
colcon build --symlink-install
source install/setup.bash
```

## Train
```bash
scripts/train.sh ddpg        # DDPG with the frozen reference config
scripts/train.sh td3
scripts/train.sh sac
```
Each run prints a live line per episode and writes its logs/plots:
```
Epi 1240   | SUCCESS     | steps 87   | total 1.2M  |  4.3s | MA100  62%
```
- console log: `log/<algo>_base_<timestamp>.log`
- metrics + plots: `src/turtlebot3_drl/model/<host>/<algo>_<i>_stage_9/` →
  `_metrics.tsv` and `training.png` (success MA100, validation success, actor/critic loss,
  reward components). Regenerate a plot anytime: `scripts/plot.sh <session_dir>`.

## Evaluate (the one benchmark)
Every checkpoint — ours or the reference's — is scored the same way: N random-goal
deterministic episodes via the reference `test_agent`.
```bash
scripts/eval.sh <algo> <model_dir> <episode> [N=100]
# e.g. scripts/eval.sh ddpg src/turtlebot3_drl/model/<host>/ddpg_3_stage_9 5000
```
Prints a summary and writes it to `eval/results/`. Reference baselines measured by this
harness: **DDPG 84% · TD3 74% · SAC 82%**. See `eval/README.md` for details.

## Try an improvement
Each algorithm folder has an `experiments/` directory. Add a file that exports only the knobs
you change, then train with it and score against the baseline:
```bash
echo 'export DRL_PER=1' > DDPG/experiments/per.sh
scripts/train.sh ddpg per
scripts/eval.sh  ddpg <model_dir> <episode>
```
