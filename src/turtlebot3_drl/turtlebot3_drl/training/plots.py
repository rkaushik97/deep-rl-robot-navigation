"""Render the 4-panel training figure from a run's logs. Reads only the plain-text
`_metrics.tsv` (+ `_eval_stage9.tsv` if present) so it can be run standalone on any
session dir: `python3 -m turtlebot3_drl.training.plots <session_dir>`.

Panels: (1) 100-ep success moving average, (2) validation success on the eval set,
(3) actor & critic loss, (4) reward components.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..drl_environment.reward import REWARD_COMPONENT_NAMES


def _read_metrics(path):
    cols = None
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if cols is None:
                cols = parts
                continue
            if parts and parts[0].isdigit():
                rows.append(parts)
    data = {c: [] for c in cols}
    for r in rows:
        for c, v in zip(cols, r):
            data[c].append(v)
    return data


def _f(xs):
    return np.array([float(x) for x in xs], dtype=float)


def draw(session_dir):
    metrics_path = os.path.join(session_dir, '_metrics.tsv')
    if not os.path.exists(metrics_path):
        return None
    d = _read_metrics(metrics_path)
    if not d.get('episode'):
        return None
    ep = _f(d['episode'])

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    # (1) success moving average
    ax[0, 0].plot(ep, _f(d['ma100_success']), color='#1f77b4', lw=2)
    ax[0, 0].set_title('success rate (100-ep moving avg)')
    ax[0, 0].set_xlabel('episode'); ax[0, 0].set_ylabel('%'); ax[0, 0].set_ylim(0, 100)
    ax[0, 0].grid(alpha=0.3)

    # (2) validation success on the eval set
    ax[0, 1].set_title('validation success (eval set)')
    ax[0, 1].set_xlabel('episode'); ax[0, 1].set_ylabel('%'); ax[0, 1].set_ylim(0, 100)
    ax[0, 1].grid(alpha=0.3)
    eval_path = os.path.join(session_dir, '_eval_stage9.tsv')
    if os.path.exists(eval_path):
        ve, vs = [], []
        for line in open(eval_path):
            p = line.split('\t')
            if p and p[0].isdigit():
                ve.append(float(p[0])); vs.append(float(p[3]) * 100.0)
        if ve:
            ax[0, 1].plot(ve, vs, 'o-', color='#2ca02c', lw=2, ms=4)
    else:
        ax[0, 1].text(0.5, 0.5, 'no validation data\n(DRL_VAL_EPS=0)', ha='center',
                      va='center', transform=ax[0, 1].transAxes, color='gray')

    # (3) actor & critic loss
    ax[1, 0].plot(ep, _f(d['loss_critic']), color='#d62728', lw=1.2, label='critic')
    ax[1, 0].plot(ep, _f(d['loss_actor']), color='#9467bd', lw=1.2, label='actor')
    ax[1, 0].set_title('avg loss per episode'); ax[1, 0].set_xlabel('episode')
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

    # (4) reward components (episode sums)
    for name in REWARD_COMPONENT_NAMES:
        if name in d:
            ax[1, 1].plot(ep, _f(d[name]), lw=1.0, label=name)
    ax[1, 1].set_title('reward components (episode sum)'); ax[1, 1].set_xlabel('episode')
    ax[1, 1].legend(fontsize=7, ncol=2); ax[1, 1].grid(alpha=0.3)

    out = os.path.join(session_dir, 'training.png')
    fig.suptitle(f'{os.path.basename(session_dir.rstrip("/"))} — {int(ep[-1])} episodes, '
                 f'last MA100 {float(d["ma100_success"][-1]):.0f}%')
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


if __name__ == '__main__':
    print(draw(sys.argv[1]))
