# -*- coding: utf-8 -*-
# @Date       : 2024/5/25 11:11
# @Auther     : Wang.zr
# @File name  : args_config.py
# @Description:
import argparse
from utils.logger import set_logger

logger = set_logger(__name__, output_file="args_config.log")


def parse_args_maddpg():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--tau", type=float, default=0.01, help="target smoothing coefficient")
    # Checkpointing
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--exp-name", type=str, default="maddpg", help="name of the experiment")
    parser.add_argument("--save-model-dir", type=str, default="../models/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="../results/maddpg/learning_curves/",
                        help="directory where plot data is saved")
    parser.add_argument("--show-plots", type=bool, default=True, help="show plots")
    args = parser.parse_args()

    # Log the parsed arguments
    logger.info("============================== MADDPG Global arguments===============================")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("=====================================================================================")
    return args


