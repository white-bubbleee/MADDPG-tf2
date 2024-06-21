import argparse
import numpy as np

import time
import pickle
import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from maddpg.trainer.maddpg import MADDPGTrainer
from base.trainer import MultiTrainerContainer
from base.args_config import get_config

from utils.utils import save_model, load_model, load_data2_plot
from utils.logger import set_logger

logger = set_logger(__name__, output_file="train_maddpg.log")


# 在任何使用到的地方
# logger.info("Start training")


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    trainer = MADDPGTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy == 'ddpg')))
    return trainers


def train(arglist):
    # Create environment
    logger.info("=====================================================================================")
    curtime = datetime.datetime.now()
    cur_dir = f"{curtime.strftime('%Y-%m-%d-%H-%M-%S')}"
    logger.info(f"Training start at {cur_dir}")
    arglist.save_dir = arglist.save_dir + arglist.exp_name + '/' + arglist.scenario + '/' + cur_dir
    logger.info(f"Save dir: {arglist.save_dir}")
    if not os.path.exists(arglist.save_dir):
        os.makedirs(arglist.save_dir)

    env = make_env(arglist.scenario, arglist, arglist.benchmark)
    # Create agent trainers
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    num_adversaries = min(env.n, arglist.num_adversaries)
    trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
    logger.info('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

    # 定义检查点，包含多个模型
    # 创建MultiAgentContainer对象
    multi_agent_container = MultiTrainerContainer(trainers)
    checkpoint = tf.train.Checkpoint(multi_agent_container=multi_agent_container)

    # Load previous results, if necessary
    if arglist.load_dir == "":
        arglist.load_dir = arglist.save_dir
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, arglist.load_dir, max_to_keep=5)
    if arglist.display or arglist.restore or arglist.benchmark:
        logger.info('Loading previous state...')
        checkpoint.restore(checkpoint_manager.latest_checkpoint)

    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve
    agent_info = [[[]]]  # placeholder for benchmarking info

    obs_n = env.reset()
    episode_step = 0
    train_step = 0
    t_start = time.time()

    logger.info('Starting iterations...')
    while True:
        # get action
        action_n = [trainer.get_action(np.expand_dims(obs, axis=0))[0] for trainer, obs in zip(trainers, obs_n)]
        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        episode_step += 1
        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)
        # collect experience
        for i, agent in enumerate(trainers):
            agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
        obs_n = new_obs_n

        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        if done or terminal:
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)
            agent_info.append([[]])

        # increment global step counter
        train_step += 1

        # for benchmarking learned policies
        if arglist.benchmark:
            for i, info in enumerate(info_n):
                agent_info[-1][i].append(info_n['n'])
            if train_step > arglist.benchmark_iters and (done or terminal):
                file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                logger.info('Finished benchmarking, now saving...')
                with open(file_name, 'wb') as fp:
                    pickle.dump(agent_info[:-1], fp)
                break
            continue

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.1)
            env.render()
            continue

        # update all trainers, if not in display or benchmark mode
        loss = None
        for agent in trainers:
            agent.pretrain()
        for agent in trainers:
            loss = agent.train(trainers, train_step)  # sample index is same.

        # save model, display training output
        if terminal and (len(episode_rewards) % arglist.save_rate == 0):
            checkpoint_manager.save()
            # print statement depends on whether or not there are adversaries
            if num_adversaries == 0:
                logger.info("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                    round(time.time() - t_start, 3)))
            else:
                logger.info(
                    "steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time() - t_start, 3)))
            t_start = time.time()
            # Keep track of final episode reward
            final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

        # saves final episode reward for plotting training curve later
        if len(episode_rewards) > arglist.num_episodes:
            file_dir = arglist.plots_dir + cur_dir + '/'
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            rew_file_name = file_dir + arglist.exp_name + '_rewards.pkl'
            with open(rew_file_name, 'wb') as fp:
                pickle.dump(final_ep_rewards, fp)
            agrew_file_name = file_dir + arglist.exp_name + '_agrewards.pkl'
            with open(agrew_file_name, 'wb') as fp:
                pickle.dump(final_ep_ag_rewards, fp)
            logger.info('...Finished total of {} episodes.'.format(len(episode_rewards)))

            if arglist.show_plots:
                load_data2_plot(rew_file_name, "reward", False)
                load_data2_plot(agrew_file_name, "agreward", False)
            break


if __name__ == '__main__':
    arglist = get_config('maddpg')
    train(arglist)
