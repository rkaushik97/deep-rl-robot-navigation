"""Universal training harness — live display, per-episode metrics, and plots.

Algorithm-agnostic: any training loop that records episodes through TrainingMetrics
gets the same on-screen line (with a 100-episode success moving average), the same
`_metrics.tsv` log, and the same multi-panel `training.png`.
"""
