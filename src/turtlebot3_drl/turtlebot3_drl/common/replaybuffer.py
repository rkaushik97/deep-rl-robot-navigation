import numpy as np
import random
from collections import deque
import itertools


class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.max_size = size

    def sample(self, batchsize):
        batch = []
        batchsize = min(batchsize, self.get_length())
        batch = random.sample(self.buffer, batchsize)
        s_array = np.float32([array[0] for array in batch])
        a_array = np.float32([array[1] for array in batch])
        r_array = np.float32([array[2] for array in batch])
        new_s_array = np.float32([array[3] for array in batch])
        done_array = np.float32([array[4] for array in batch])

        return s_array, a_array, r_array, new_s_array, done_array

    def get_length(self):
        return len(self.buffer)

    def add_sample(self, s, a, r, new_s, done):
        transition = (s, a, r, new_s, done)
        self.buffer.append(transition)


class _SumTree:
    """Fixed-capacity sum-tree: O(log n) prioritized sampling + priority updates."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        if left >= len(self.tree):          # leaf reached
            return idx
        right = left + 1
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p):
        idx = self.write + self.capacity - 1
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1
        return idx

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        return idx, self.tree[idx]


class PrioritizedReplayBuffer:
    """Proportional Prioritized Experience Replay (Schaul et al. 2016).

    Drop-in for ReplayBuffer (same add_sample/sample/get_length + a .buffer deque so
    the existing save/load-buffer path still works), but sample() also returns the
    sampled tree indices and importance-sampling (IS) weights, and update_priorities()
    re-weights transitions by their latest |TD error|. Gated by DRL_PER=1.
    """
    def __init__(self, size, alpha=0.6, beta=0.4, beta_increment=1e-6, eps=1e-3):
        self.max_size = size
        self.tree = _SumTree(size)
        self.data = [None] * size
        self.alpha = alpha                  # priority exponent (0=uniform, 1=full)
        self.beta = beta                    # IS-correction exponent, annealed -> 1
        self.beta_increment = beta_increment
        self.eps = eps                      # floor so no transition has 0 probability
        self.max_priority = 1.0
        self.buffer = deque(maxlen=size)    # compat shim for save/load + get_length fallback

    def get_length(self):
        return self.tree.n_entries

    def add_sample(self, s, a, r, new_s, done):
        transition = (s, a, r, new_s, done)
        idx = self.tree.add(self.max_priority ** self.alpha)   # new samples get max priority
        self.data[idx - (self.max_size - 1)] = transition
        self.buffer.append(transition)

    def sample(self, batchsize):
        batchsize = min(batchsize, self.get_length())
        batch, idxs, priorities = [], [], []
        segment = self.tree.total() / batchsize
        self.beta = min(1.0, self.beta + self.beta_increment)
        for i in range(batchsize):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p = self.tree.get(s)
            data = self.data[idx - (self.max_size - 1)]
            if data is None:                # rare edge: resample uniformly
                data = random.choice([d for d in self.data if d is not None])
            batch.append(data); idxs.append(idx); priorities.append(p)
        probs = np.array(priorities) / (self.tree.total() + 1e-12)
        is_weights = (self.get_length() * probs + 1e-12) ** (-self.beta)
        is_weights = (is_weights / is_weights.max()).astype(np.float32)
        s_array = np.float32([b[0] for b in batch])
        a_array = np.float32([b[1] for b in batch])
        r_array = np.float32([b[2] for b in batch])
        new_s_array = np.float32([b[3] for b in batch])
        done_array = np.float32([b[4] for b in batch])
        return s_array, a_array, r_array, new_s_array, done_array, np.array(idxs), is_weights

    def update_priorities(self, idxs, td_errors):
        for idx, td in zip(idxs, td_errors):
            mag = float(abs(td)) + self.eps
            self.tree.update(idx, mag ** self.alpha)
            if mag > self.max_priority:
                self.max_priority = mag
