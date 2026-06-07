#!/usr/bin/env python3
# Build a side-by-side GIF from a capture .npz: LEFT = gazebo top-down frame, RIGHT = the agent's
# network visualization (states, hidden layer 0/1, action linear/angular, accumulated reward) —
# tomasvr-style. Headless (Agg).
#   python3 scripts/make_visual_gif.py <capture.npz> <out.gif> [fps]
import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, PillowWriter

d = np.load(sys.argv[1])
out = sys.argv[2]
fps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
states, h0, h1, act, cr, frames = d['states'], d['h0'], d['h1'], d['actions'], d['cumreward'], d['frames']
n = len(states)
h_max = max(float(h0.max()), float(h1.max()), 1e-3)
cr_max = max(float(cr.max()) * 1.1, 100)

plt.rcParams.update({'axes.facecolor': '#111', 'figure.facecolor': '#111', 'text.color': '#ddd',
                     'axes.labelcolor': '#ddd', 'xtick.color': '#888', 'ytick.color': '#888',
                     'axes.titlecolor': '#eee', 'axes.edgecolor': '#444'})
fig = plt.figure(figsize=(12, 6))
gs = GridSpec(4, 6, figure=fig, wspace=0.35, hspace=0.7, left=0.03, right=0.98, top=0.93, bottom=0.08)
ax_g = fig.add_subplot(gs[:, 0:3]); ax_g.axis('off'); ax_g.set_title('Gazebo — stage 9', fontsize=11)
ax_s = fig.add_subplot(gs[0, 3:6])
ax_0 = fig.add_subplot(gs[1, 3:6])
ax_1 = fig.add_subplot(gs[2, 3:6])
ax_l = fig.add_subplot(gs[3, 3])
ax_a = fig.add_subplot(gs[3, 4])
ax_r = fig.add_subplot(gs[3, 5])
for ax in (ax_s, ax_0, ax_1, ax_l, ax_a, ax_r):
    ax.tick_params(labelsize=6)


def draw(f):
    ax_g.clear(); ax_g.axis('off'); ax_g.set_title('Gazebo — stage 9', fontsize=11)
    ax_g.imshow(frames[f])
    ax_s.clear(); ax_s.bar(range(states.shape[1]), states[f], color='#4FC3F7', width=1.0)
    ax_s.set_ylim(-1, 1); ax_s.set_title('States (40 lidar + goal + prev-action)', fontsize=9); ax_s.set_xticks([])
    ax_0.clear(); ax_0.bar(range(h0.shape[1]), h0[f], color='#81C784', width=1.0)
    ax_0.set_ylim(0, h_max); ax_0.set_title('Hidden layer 0', fontsize=9); ax_0.set_xticks([])
    ax_1.clear(); ax_1.bar(range(h1.shape[1]), h1[f], color='#FFB74D', width=1.0)
    ax_1.set_ylim(0, h_max); ax_1.set_title('Hidden layer 1', fontsize=9); ax_1.set_xticks([])
    ax_l.clear(); ax_l.bar([0], [act[f, 0]], color='#E57373', width=0.6)
    ax_l.set_ylim(-1, 1); ax_l.set_title('Action\nlinear', fontsize=8); ax_l.set_xticks([])
    ax_a.clear(); ax_a.bar([0], [act[f, 1]], color='#BA68C8', width=0.6)
    ax_a.set_ylim(-1, 1); ax_a.set_title('Action\nangular', fontsize=8); ax_a.set_xticks([])
    ax_r.clear(); ax_r.bar([0], [cr[f]], color='#FFD54F', width=0.6)
    ax_r.set_ylim(0, cr_max); ax_r.set_title('Accum.\nreward', fontsize=8); ax_r.set_xticks([])
    fig.suptitle(f'DDPG navigating to goal  —  step {f+1}/{n}', fontsize=12, color='#fff')


anim = FuncAnimation(fig, draw, frames=n, interval=1000 / fps)
anim.save(out, writer=PillowWriter(fps=fps))
print(f'gif -> {out}  ({n} frames @ {fps}fps)')
