"""Per-episode training metrics: 100-episode success moving average, reward-component
sums, and losses, persisted to `<session>/_metrics.tsv` (one row per episode).

This TSV is the single source the plots read, and is trivial to inspect by hand.
"""
import os
from collections import deque

from ..drl_environment.reward import REWARD_COMPONENT_NAMES
from ..common.settings import SUCCESS

# _metrics.tsv columns (tab-separated). Reward components are episode SUMS.
COLUMNS = (['episode', 'total_steps', 'outcome', 'reward_sum', 'ma100_success',
            'loss_critic', 'loss_actor'] + list(REWARD_COMPONENT_NAMES))


class TrainingMetrics:
    def __init__(self, session_dir, window=100):
        self.path = os.path.join(session_dir, '_metrics.tsv')
        self.window = window
        self.outcomes = deque(maxlen=window)
        # On resume, rebuild the MA window from any existing rows so the moving
        # average is continuous rather than restarting at the resume point.
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    rows = [l.split('\t') for l in f if l and l[0].isdigit()]
                for r in rows[-window:]:
                    self.outcomes.append(1 if int(r[2]) == SUCCESS else 0)
            except (ValueError, IndexError):
                pass
        else:
            with open(self.path, 'w') as f:
                f.write('\t'.join(COLUMNS) + '\n')

    def ma100(self):
        """Success rate (%) over the last `window` episodes."""
        return 100.0 * sum(self.outcomes) / len(self.outcomes) if self.outcomes else 0.0

    def record(self, episode, total_steps, outcome, reward_sum,
               loss_critic_avg, loss_actor_avg, component_sums):
        """Append one episode and return the updated MA100 success rate."""
        self.outcomes.append(1 if outcome == SUCCESS else 0)
        ma = self.ma100()
        comps = list(component_sums) + [0.0] * (len(REWARD_COMPONENT_NAMES) - len(component_sums))
        row = [episode, total_steps, outcome, f'{reward_sum:.3f}', f'{ma:.2f}',
               f'{loss_critic_avg:.5f}', f'{loss_actor_avg:.5f}'] + [f'{c:.3f}' for c in comps]
        with open(self.path, 'a') as f:
            f.write('\t'.join(str(x) for x in row) + '\n')
        return ma
