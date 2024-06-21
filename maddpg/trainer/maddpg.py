# -*- coding: utf-8 -*-
# @Date       : 2024/5/2219:13
# @Auther     : Wang.zr
# @File name  : maddpg.py
# @Description:

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model
from base.replaybuffer import ReplayBuffer
from base.trainer import ACAgent, Trainer
import numpy as np
import gym
from ..common.distribution import gen_action_for_discrete, gen_action_for_continuous
from utils.logger import set_logger

logger = set_logger(__name__, output_file="maddpg.log")
# 在任何使用到的地方
# logger.info("Start training")

DATA_TYPE = tf.float64


class MADDPGAgent(ACAgent):
    def __init__(self, name, action_dim, obs_dim, agent_index, args, local_q_func=False):
        super().__init__(name, action_dim, obs_dim, agent_index, args)
        self.name = name + "_agent_" + str(agent_index)
        self.act_dim = action_dim[agent_index]
        self.obs_dim = obs_dim[agent_index][0]

        self.act_total = sum(action_dim)
        self.obs_total = sum([obs_dim[i][0] for i in range(len(obs_dim))])

        self.num_units = args.num_units
        self.local_q_func = local_q_func
        self.nums_agents = len(action_dim)
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        self.target_actor = self.build_actor()
        self.target_critic = self.build_critic()

        self.actor_optimizer = tf.keras.optimizers.Adam(args.lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(args.lr)

    def build_actor(self, action_bound=None):
        obs_input = Input(shape=(self.obs_dim,))
        out = Dense(self.num_units, activation='relu')(obs_input)
        out = Dense(self.num_units, activation='relu')(out)
        out = Dense(self.act_dim, activation=None)(out)
        out = tf.cast(out, DATA_TYPE)

        actor = Model(inputs=obs_input, outputs=out)

        return actor

    def build_critic(self):
        # ddpg or maddpg
        if self.local_q_func:  # ddpg
            obs_input = Input(shape=(self.obs_dim,))
            act_input = Input(shape=(self.act_dim,))
            concatenated = Concatenate(axis=1)([obs_input, act_input])
        if not self.local_q_func:  # maddpg
            obs_input_list = [Input(shape=(self.obs_dim,)) for _ in range(self.nums_agents)]
            act_input_list = [Input(shape=(self.act_dim,)) for _ in range(self.nums_agents)]
            concatenated_obs = Concatenate(axis=1)(obs_input_list)
            concatenated_act = Concatenate(axis=1)(act_input_list)
            concatenated = Concatenate(axis=1)([concatenated_obs, concatenated_act])
        out = Dense(self.num_units, activation='relu')(concatenated)
        out = Dense(self.num_units, activation='relu')(out)
        out = Dense(1, activation=None)(out)
        out = tf.cast(out, DATA_TYPE)

        critic = Model(inputs=obs_input_list + act_input_list if not self.local_q_func else [obs_input, act_input],
                       outputs=out)

        return critic

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=DATA_TYPE)])
    def agent_action(self, obs):
        return self.actor(obs)

    @tf.function
    def agent_critic(self, obs_act):
        return self.critic(obs_act)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=DATA_TYPE)])
    def agent_target_action(self, obs):
        return self.target_actor(obs)

    @tf.function
    def agent_target_critic(self, obs_act):
        return self.target_critic(obs_act)

    def save_model(self, path):
        actor_path = f"{path}_{self.name}_actor.h5"
        critic_path = f"{path}_{self.name}_critic.h5"

        self.actor.save(actor_path)
        self.critic.save(critic_path)

        print(f"Actor model saved at {actor_path}")
        print(f"Critic model saved at {critic_path}")

    def load_model(self, path):
        actor_path = f"{path}_{self.name}_actor.h5"
        critic_path = f"{path}_{self.name}_critic.h5"

        self.actor = tf.keras.models.load_model(actor_path)
        self.critic = tf.keras.models.load_model(critic_path)

        print(f"Actor model loaded from {actor_path}")
        print(f"Critic model loaded from {critic_path}")


class MADDPGTrainer(Trainer):
    def __init__(self, name, obs_dims, action_space, agent_index, args, local_q_func=False):
        super().__init__(name, obs_dims, action_space, agent_index, args, local_q_func)
        self.name = name
        self.args = args
        self.agent_index = agent_index
        self.nums = len(obs_dims)

        # ======================= env preprocess =========================
        self.action_space = action_space
        if isinstance(action_space[0], gym.spaces.Box):
            self.act_dims = [self.action_space[i].shape[0] for i in range(self.nums)]
            self.action_out_func = gen_action_for_continuous
        elif isinstance(action_space[0], gym.spaces.Discrete):
            self.act_dims = [self.action_space[i].n for i in range(self.nums)]
            self.action_out_func = gen_action_for_discrete

        # ====================== hyperparameters =========================
        self.local_q_func = local_q_func
        if self.local_q_func:
            logger.info(f"Init {agent_index} is using DDPG algorithm")
        else:
            logger.info(f"Init {agent_index} is using MADDPG algorithm")
        self.gamma = args.gamma
        self.tau = args.tau
        self.batch_size = args.batch_size

        self.agent = MADDPGAgent(name, self.act_dims, obs_dims, agent_index, args, local_q_func=local_q_func)
        self.replay_buffer = ReplayBuffer(args.buffer_size)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

        # ====================initialize target networks====================
        self.update_target(self.agent.target_actor.variables, self.agent.actor.variables, tau=self.tau)
        self.update_target(self.agent.target_critic.variables, self.agent.critic.variables, tau=self.tau)

    def train(self, trainers, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len:  # replay buffer is not large enough
            return

        if not t % 100 == 0:  # only update every 100 steps
            return

        obs_n, action_n, reward_i, next_obs_n, done_i = self.sample_batch_for_pretrain(trainers)
        # ======================== train critic ==========================
        with tf.GradientTape() as tape:
            target_actions = [trainer.get_target_action(next_obs_n[i]) for i, trainer in enumerate(trainers)]
            #  ============= target ===========
            target_q_input = next_obs_n + target_actions  # global info
            if self.local_q_func:
                target_q_input = [next_obs_n[self.agent_index], target_actions[self.agent_index]]
            target_q = self.agent.agent_target_critic(target_q_input)
            # done_i = tf.convert_to_tensor(done_i[:, np.newaxis])
            # done_i = done_i[:, np.newaxis]
            # reward_i = reward_i[:, np.newaxis]
            # reward_i = tf.convert_to_tensor(reward_i[:, np.newaxis])
            y = reward_i + self.gamma * (1 - done_i) * target_q  # target

            # ============= current ===========
            q_input = obs_n + action_n  # global info
            if self.local_q_func:  # local info
                q_input = [obs_n[self.agent_index], action_n[self.agent_index]]
            q = self.agent.agent_critic(q_input)
            critic_loss = tf.reduce_mean(tf.square(y - q))
        critic_grads = tape.gradient(critic_loss, self.agent.critic.trainable_variables)
        self.agent.critic_optimizer.apply_gradients(zip(critic_grads, self.agent.critic.trainable_variables))

        # ========================= train actor ===========================
        with tf.GradientTape() as tape:
            _action_n = []
            for i, trainer in enumerate(trainers):
                _action = trainer.get_action(obs_n[i])
                _action_n.append(_action)
            q_input = obs_n + _action_n
            if self.local_q_func:
                q_input = [obs_n[self.agent_index], _action_n[self.agent_index]]
            p_reg = tf.reduce_mean(tf.square(_action_n[self.agent_index]))  # regularization
            actor_loss = -tf.reduce_mean(self.agent.agent_critic(q_input)) + p_reg * 1e-3

        actor_grads = tape.gradient(actor_loss, self.agent.actor.trainable_variables)
        self.agent.actor_optimizer.apply_gradients(zip(actor_grads, self.agent.actor.trainable_variables))

        # ======================= update target networks ===================
        self.update_target(self.agent.target_actor.variables, self.agent.actor.variables, self.tau)
        self.update_target(self.agent.target_critic.variables, self.agent.critic.variables, self.tau)

    def pretrain(self):
        self.replay_sample_index = None

    def save_model(self, path):
        checkpoint = tf.train.Checkpoint(agents=self.agent)
        checkpoint.save(path)

    def locd_model(self, path):
        self.agent.load_model(path)

    @tf.function
    def get_action(self, state):
        # return tf.cond(
        #     tf.rank(state) == 1,
        #     lambda: self.action_out_func(self.agent.agent_action(state.squeeze(axis=0))[0]),
        #     lambda: self.action_out_func(self.agent.agent_action(state))
        # )
        # if state.ndim == 1:
        #     state = np.expand_dims(state, axis=0)
        #     action_re = self.action_out_func(self.agent.actor(state)[0])
        # else:
        #     action_re = self.action_out_func(self.agent.actor(state))
        return self.action_out_func(self.agent.actor(state))

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=DATA_TYPE)])
    def get_target_action(self, state):
        return self.action_out_func(self.agent.target_actor(state))

    def update_target(self, target_weights, weights, tau):
        for (target, weight) in zip(target_weights, weights):
            target.assign(weight * tau + target * (1 - tau))

    def experience(self, state, action, reward, next_state, done, terminal):
        self.replay_buffer.add(state, action, reward, next_state, float(done))

    def sample_batch_for_pretrain(self, trainers):
        if self.replay_sample_index is None:
            self.replay_sample_index = self.replay_buffer.make_index(self.batch_size)
        obs_n, action_n, next_obs_n = [], [], []
        reward_i, done_i = None, None
        for i, trainer in enumerate(trainers):
            obs, act, rew, next_obs, done = trainer.replay_buffer.sample_index(self.replay_sample_index)

            # obs = tf.convert_to_tensor(obs, dtype=tf.float64)               # (self.batch_size, 18)
            # act = tf.convert_to_tensor(act, dtype=tf.float64)               # (self.batch_size, 5)
            # rew = tf.convert_to_tensor(rew, dtype=tf.float64)               # (self.batch_size, 1)
            # next_obs = tf.convert_to_tensor(next_obs, dtype=tf.float64)     # (self.batch_size, 18)
            # done = tf.convert_to_tensor(done, dtype=tf.float64)             # (self.batch_size, 1)

            # obs = np.array(obs)               # (self.batch_size, 18)
            # act = np.array(act)               # (self.batch_size, 5)
            # rew = np.array(rew)                # (self.batch_size, 1)
            # next_obs = np.array(next_obs)      # (self.batch_size, 18)
            # done = np.array(done)              # (self.batch_size, 1)

            obs_n.append(obs)
            action_n.append(act)
            next_obs_n.append(next_obs)

            if self.agent_index == i:
                done_i = done
                reward_i = rew
        return obs_n, action_n, reward_i[:, np.newaxis], next_obs_n, done_i[:, np.newaxis]
