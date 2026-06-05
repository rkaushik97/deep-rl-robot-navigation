# Legacy scripts (retired)

Superseded by `scripts/train.sh`, `scripts/eval.sh`, `scripts/plot.sh` and the in-package
`training/` + `evaluation/` modules. Kept for reference only — not maintained.

- `clean_eval.sh`, `replay_analyze.py` — the old custom 40-goal fixed-list benchmark. **Do not
  use** — it is NOT the reference eval. Use `scripts/eval.sh` (reference `test_agent`).
- `standard_eval.sh` — ran the reference repo's `test_agent` out-of-tree; replaced by the native
  `scripts/eval.sh`.
- `plot_val.py`, `plot_replication.py`, `plot_experiments.py`, `_plot_loop.sh` — replaced by
  `training/plots.py` (+ `scripts/plot.sh`).
- `run_experiment.sh`, `resume_experiment.sh`, `run_single.sh`, `analyze_experiment.py`,
  `push_experiment.sh`, `_wait_td3eval.sh` — ad-hoc experiment plumbing, replaced by
  `scripts/train.sh` + per-algorithm `config.sh`.
