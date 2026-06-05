import numpy as np

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.99, min_sigma=0.01, decay_period=600000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_noise(self, t=0):
        # Linear anneal: sigma = max at t=0 -> min at t>=decay_period (t is the GLOBAL step).
        # (Was a cumulative per-call subtraction that collapsed to min in ~6 steps.)
        frac = min(1.0, max(0.0, float(t)) / self.decay_period)
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * frac
        return self.evolve_state()
