# -*- coding: utf-8 -*-
# @Date       : 2024/5/2310:02
# @Auther     : Wang.zr
# @File name  : trainer.py
# @Description:
import tensorflow as tf


class ACAgent:

    def __init__(self, name, action_dim, obs_dim, agent_index, args):
        pass

    def build_actor(self):
        raise NotImplementedError

    def build_critic(self):
        raise NotImplementedError



class Trainer:

    def __init__(self, name, obs_dims, action_space, agent_index, args, local_q_func=False):
        self.name = name
        pass

    def get_action(self, state):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def sample_batch_for_pretrain(self, trainers):
        raise NotImplementedError

    def pretrain(self):
        raise NotImplementedError

    def update_target(self, target_weights, weights, tau):
        for (target, weight) in zip(target_weights, weights):
            target.assign(weight * tau + target * (1 - tau))


class MultiTrainerContainer(tf.train.Checkpoint):

    def __init__(self, trainers):
        super().__init__()
        self.trainer = {trainer.name: trainer for trainer in trainers}
