#!/usr/bin/env python3
# Analyze a tracked experiment's training log -> experiments/<EXP>/analysis.md
#   python3 scripts/analyze_experiment.py <EXP_NAME>
# Parses the per-episode 'Epi: ... outcome:' lines: success-rate curve, failure-mode
# breakdown (where it's lacking), trend, and a plain verdict to guide the next step.
import sys, os, re

BASE = '/home/kaushik/project/deep-rl-robot-navigation'
exp = sys.argv[1]
expdir = os.path.join(BASE, 'experiments', exp)
log = os.path.join(expdir, 'train.log')
if not os.path.exists(log):
    sys.exit(f"no train.log at {log}")

rows = []  # (episode, reward, outcome, steps)
for line in open(log):
    if not line.startswith('Epi:'):
        continue
    ep = re.search(r'Epi:\s*(\d+)', line)
    rw = re.search(r'R:\s*(-?\d+)', line)
    oc = re.search(r'outcome:\s*(\S+)', line)
    st = re.search(r'steps:\s*(\d+)', line)
    if ep and oc:
        rows.append((int(ep.group(1)), int(rw.group(1)) if rw else 0, oc.group(1), int(st.group(1)) if st else 0))

n = len(rows)
def rate(items, code): return 100.0*sum(1 for o in items if o==code)/len(items) if items else 0.0
outs = [r[2] for r in rows]
codes = ['SUCCESS','COLL_WALL','COLL_OBST','TIMEOUT','TUMBLE']

lines = [f"# Analysis: {exp}\n", f"Total training episodes: **{n}**\n"]
if n:
    last100 = outs[-100:]
    lines.append("## Headline\n")
    lines.append(f"- Success rate, **last 100 eps: {rate(last100,'SUCCESS'):.0f}%**  (overall: {rate(outs,'SUCCESS'):.0f}%)\n")
    lines.append(f"- Failure breakdown, last 100 eps: wall {rate(last100,'COLL_WALL'):.0f}% | obstacle {rate(last100,'COLL_OBST'):.0f}% | timeout {rate(last100,'TIMEOUT'):.0f}% | tumble {rate(last100,'TUMBLE'):.0f}%\n")
    # success curve in 100-ep blocks
    lines.append("\n## Success-rate curve (per 100-ep block)\n```\n")
    nb = (n + 99)//100
    for b in range(nb):
        blk = outs[b*100:(b+1)*100]
        bar = '#' * int(rate(blk,'SUCCESS')/5)
        lines.append(f"ep {b*100+1:>4}-{b*100+len(blk):<4}: {rate(blk,'SUCCESS'):>3.0f}% {bar}\n")
    lines.append("```\n")
    # failure modes over whole run
    lines.append("\n## Where it's lacking (failure modes, whole run)\n")
    for c in codes[1:]:
        lines.append(f"- {c}: {rate(outs,c):.0f}%\n")
    # trend verdict
    first = rate(outs[:max(1,n//5)],'SUCCESS'); last = rate(outs[-max(1,n//5):],'SUCCESS')
    lines.append("\n## Verdict\n")
    if last > first + 10:
        v = f"LEARNING — success rose {first:.0f}% -> {last:.0f}% (first vs last fifth)."
    elif last < first - 10:
        v = f"DEGRADING — success fell {first:.0f}% -> {last:.0f}%."
    else:
        v = f"FLAT — success {first:.0f}% -> {last:.0f}% (no clear learning)."
    lines.append(f"- {v}\n")
    dom = max(codes[1:], key=lambda c: rate(last100,c))
    lines.append(f"- Dominant failure (last 100): **{dom}** ({rate(last100,dom):.0f}%) -> primary thing to address next.\n")

out = os.path.join(expdir, 'analysis.md')
open(out, 'w').write(''.join(lines))
print(''.join(lines))
print(f"\n[written] {out}")
