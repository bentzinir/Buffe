import numpy as np
import random

class ER(object):

    def __init__(self, memory_size, state_dim, reward_dim, batch_size, history_length=1):
        self.memory_size = memory_size
        self.actions = np.random.normal(scale=0.35, size=self.memory_size)
        self.rewards = np.random.normal(scale=0.35, size=(self.memory_size, reward_dim))
        self.states = np.random.normal(scale=0.35, size=(self.memory_size, state_dim))
        self.terminals = np.zeros(self.memory_size, dtype=np.float32)
        self.batch_size = batch_size
        self.history_length = history_length
        # self.count = 0
        self.count = 200
        self.current = 0

        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty((self.batch_size, self.history_length, state_dim), dtype = np.float32)
        self.poststates = np.empty((self.batch_size, self.history_length, state_dim), dtype = np.float32)

    def add(self, actions, rewards, states, terminals):
        # NB! state is post-state, after action and reward
        for a, r, s, t in zip(actions, rewards, states, terminals):
            self.actions[self.current] = a
            self.rewards[self.current, ...] = r
            self.states[self.current, ...] = s
            self.terminals[self.current] = t
            self.count = max(self.count, self.current + 1)
            self.current = (self.current + 1) % self.memory_size

    def get_state(self, index):
        assert self.count > 0, "replay memory is empy"
        # normalize index to expected range, allows negative indexes
        index = index % self.count
        # if is not in the beginning of matrix
        if index >= self.history_length - 1:
            # use faster slicing
            return self.states[(index - (self.history_length - 1)):(index + 1), ...]
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
            return self.states[indexes, ...]

    def sample(self):
        # memory must include poststate, prestate and history
        assert self.count > self.history_length
        # sample random indexes
        indexes = []
        while len(indexes) < self.batch_size:
            # find random index
            while True:
                # sample one index (ignore states wraping over
                index = random.randint(self.history_length, self.count - 1)
                # if wraps over current pointer, then get new one
                if index >= self.current and index - self.history_length < self.current:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self.terminals[(index - self.history_length):index].any():
                    continue
                # otherwise use this index
                break

            # NB! having index first is fastest in C-order matrices
            self.prestates[len(indexes), ...] = self.get_state(index - 1)
            self.poststates[len(indexes), ...] = self.get_state(index)
            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes, ...]
        terminals = self.terminals[indexes]

        return self.prestates, actions, rewards, self.poststates, terminals

    def save(self):
        pass

    def load(self):
        pass