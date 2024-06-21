# -*- coding: utf-8 -*-
# @Date       : 2024/5/2321:07
# @Auther     : Wang.zr
# @File name  : replaybuffer.py
# @Description:
from collections import deque
import tensorflow as tf
import random
import numpy as np


class ReplayBufferTensor(object):

    def __init__(self, size):
        self.deque = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        state = tf.cast(state, tf.float64)
        action = tf.cast(action, tf.float64)
        reward = tf.cast(reward, tf.float64)
        next_state = tf.cast(next_state, tf.float64)
        done = tf.cast(done, tf.float64)
        state = tf.expand_dims(state, axis=0)
        action = tf.expand_dims(action, axis=0)
        reward = tf.expand_dims(reward, axis=0)
        next_state = tf.expand_dims(next_state, axis=0)

        done = tf.expand_dims(done, axis=0)
        self.deque.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.deque, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones

    def sample_from_index(self, index):
        """根据给定的索引列表,从 replay buffer 中采样数据"""
        states = tf.concat([self.deque[i][0] for i in index], axis=0)
        actions = tf.concat([self.deque[i][1] for i in index], axis=0)
        rewards = tf.concat([self.deque[i][2] for i in index], axis=0)
        next_states = tf.concat([self.deque[i][3] for i in index], axis=0)
        dones = tf.concat([self.deque[i][4] for i in index], axis=0)

        return states, actions, rewards, next_states, dones

    def gen_index(self, batch_size):
        return random.sample(range(len(self.deque)), batch_size)

    def __len__(self):
        return len(self.deque)


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)


def create_replaybuffer_from_tf(size):
    pass
