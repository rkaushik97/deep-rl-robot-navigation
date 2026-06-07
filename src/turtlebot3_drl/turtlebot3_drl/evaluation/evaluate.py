"""Read a `test_agent` log and print the standard evaluation summary.

The test log's last data row carries cumulative counts in the final column as
`success/wall/obstacle/timeout/tumble`. This module turns that into a one-screen
summary and (optionally) copies the result into evaluation/results/.

Standalone:  python3 -m turtlebot3_drl.evaluation.evaluate <test_log.txt> [label]
"""
import os
import sys


SUCCESS = 1  # outcome codes (mirror common.settings; kept local so evaluate.py stays import-light)


def parse(test_log_path):
    """Parse a test log into outcome counts + derived navigation metrics.

    Returns dict(success, wall, obstacle, timeout, tumble, episodes, collision_rate,
    time_to_goal, path_efficiency). The last three are means over SUCCESSFUL episodes
    (time_to_goal in s; path_efficiency = straight-line/actual path, in (0, 1]).
    Per-episode rows: 'entry, outcome, step, duration, distance, initial_distance, counts'.
    Older logs without the initial_distance column still parse (path_efficiency -> None)."""
    last_counts = None
    durations, path_effs = [], []
    with open(test_log_path) as f:
        for line in f:
            line = line.strip()
            if not (line and line[0].isdigit()):
                continue
            tok = [t.strip() for t in line.split(',')]
            last_counts = tok[-1]                         # cumulative "s/cw/co/t/tu"
            try:
                outcome = int(tok[1])
            except (ValueError, IndexError):
                continue
            if outcome == SUCCESS and len(tok) >= 5:
                durations.append(float(tok[3]))
                if len(tok) >= 7:                        # new format carries initial_distance
                    dist, init = float(tok[4]), float(tok[5])
                    if dist > 0:
                        path_effs.append(min(1.0, init / dist))
    if last_counts is None:
        return None
    parts = [int(x) for x in last_counts.split('/')]
    parts += [0] * (5 - len(parts))
    s, cw, co, t, tu = parts[:5]
    n = s + cw + co + t + tu
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    return dict(success=s, wall=cw, obstacle=co, timeout=t, tumble=tu, episodes=n,
                collision_rate=(cw + co) / n if n else 0.0,
                time_to_goal=mean(durations),
                path_efficiency=mean(path_effs))


def summarize(test_log_path, label='', results_dir=None):
    """Print the standard summary; optionally copy the log into results_dir. Returns the dict."""
    r = parse(test_log_path)
    if not r:
        print(f"[eval] no episodes logged in {test_log_path}")
        return None
    n = max(1, r['episodes'])
    pct = lambda v: f"{100 * v / n:.0f}%"
    print("\n===== EVALUATION (test_agent, random goals) =====")
    if label:
        print(f"  {label}")
    print(f"  episodes = {r['episodes']}")
    print(f"  SUCCESS  = {r['success']:3d}  ({pct(r['success'])})")
    print(f"  wall     = {r['wall']:3d}  ({pct(r['wall'])})")
    print(f"  obstacle = {r['obstacle']:3d}  ({pct(r['obstacle'])})")
    print(f"  timeout  = {r['timeout']:3d}  ({pct(r['timeout'])})")
    if r['tumble']:
        print(f"  tumble   = {r['tumble']:3d}  ({pct(r['tumble'])})")
    fmt = lambda v, suf='': f"{v:.3f}{suf}" if v is not None else "n/a"
    print(f"  collision_rate  = {r['collision_rate']*100:.0f}%   (wall+obstacle)")
    print(f"  time_to_goal    = {fmt(r['time_to_goal'], ' s')}   (mean, successes)")
    print(f"  path_efficiency = {fmt(r['path_efficiency'])}   (straight-line/actual, successes)")
    print("=================================================\n", flush=True)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        name = (label or os.path.basename(test_log_path)).replace('/', '_').replace(' ', '_')
        with open(os.path.join(results_dir, f"{name}_{n}eps.txt"), 'w') as f:
            f.write(f"{label}\nepisodes={n} success={r['success']} ({pct(r['success'])}) "
                    f"wall={r['wall']} obstacle={r['obstacle']} timeout={r['timeout']} tumble={r['tumble']}\n")
            f.write(f"collision_rate={r['collision_rate']:.4f} "
                    f"time_to_goal={fmt(r['time_to_goal'])} path_efficiency={fmt(r['path_efficiency'])}\n")
            f.write(f"source_log={test_log_path}\n")
    return r


if __name__ == '__main__':
    summarize(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '')
