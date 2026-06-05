"""The single on-screen training line, printed once per finished episode."""


def _human(n):
    n = int(n)
    if n >= 1_000_000:
        return f'{n / 1_000_000:.1f}M'
    if n >= 1_000:
        return f'{n / 1_000:.0f}k'
    return str(n)


def episode_line(episode, outcome_str, steps, total_steps, duration, ma100_success):
    """e.g. 'Epi 1234 | SUCCESS    | steps  87 | total 1.2M | 4.3s | MA100 62%'"""
    return (f'Epi {episode:<6} | {outcome_str:<11} | steps {steps:<4} | '
            f'total {_human(total_steps):<5} | {duration:4.1f}s | MA100 {ma100_success:3.0f}%')
