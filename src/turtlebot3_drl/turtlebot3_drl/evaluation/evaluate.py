"""Read a `test_agent` log and print the standard evaluation summary.

The test log's last data row carries cumulative counts in the final column as
`success/wall/obstacle/timeout/tumble`. This module turns that into a one-screen
summary and (optionally) copies the result into evaluation/results/.

Standalone:  python3 -m turtlebot3_drl.evaluation.evaluate <test_log.txt> [label]
"""
import os
import sys


def parse(test_log_path):
    """Return dict(success, wall, obstacle, timeout, tumble, episodes) from a test log."""
    last = None
    with open(test_log_path) as f:
        for line in f:
            line = line.strip()
            if line and line[0].isdigit():
                last = line
    if last is None:
        return None
    counts = last.split(',')[-1].strip()                 # "84/14/0/2/0"
    parts = [int(x) for x in counts.split('/')]
    parts += [0] * (5 - len(parts))
    s, cw, co, t, tu = parts[:5]
    n = s + cw + co + t + tu
    return dict(success=s, wall=cw, obstacle=co, timeout=t, tumble=tu, episodes=n)


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
    print("=================================================\n", flush=True)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        name = (label or os.path.basename(test_log_path)).replace('/', '_').replace(' ', '_')
        with open(os.path.join(results_dir, f"{name}_{n}eps.txt"), 'w') as f:
            f.write(f"{label}\nepisodes={n} success={r['success']} ({pct(r['success'])}) "
                    f"wall={r['wall']} obstacle={r['obstacle']} timeout={r['timeout']} tumble={r['tumble']}\n")
            f.write(f"source_log={test_log_path}\n")
    return r


if __name__ == '__main__':
    summarize(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '')
