# -*- coding: utf-8 -*-
# @Date       : 2024/5/2422:48
# @Auther     : Wang.zr
# @File name  : logger.py
# @Description:

import logging
import os
from datetime import datetime

# 定义全局的日志文件夹
LOGS_DIR = None
LOGS_DIR_INITIALIZED = False

# 定义全局的算法和场景
ALGO = "imac"
SCENARIO = "simple_spread"

QUES_FLAG = 0


def initialize_logs_dir():
    global LOGS_DIR, LOGS_DIR_INITIALIZED, QUES_FLAG, ALGO, SCENARIO

    if QUES_FLAG == 0:
        print(f"现在logs文件存储信息如下: {ALGO}/{SCENARIO}")
        print(f"是否要修改信息 y/n: ")
        if input() == "y":
            print(f"请输入算法名称 / maddpg / imac / mboma: ")
            ALGO = input()
            print("请输入场景名称 simple_spread / simple / simple_tag: ")
            SCENARIO = input()
        QUES_FLAG = 1


if not LOGS_DIR_INITIALIZED:
    LOGS_DIR = os.path.join(f"../logs/{ALGO}/{SCENARIO}", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    LOGS_DIR_INITIALIZED = True


def set_logger(name=None, output_file=None):
    global LOGS_DIR_INITIALIZED
    initialize_logs_dir()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s %(levelname)s] %(module)s:%(funcName)s:%(lineno)s: %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # file logging: all workers
    if output_file is not None:
        if output_file.endswith(".txt") or output_file.endswith(".log"):
            filename = os.path.join(LOGS_DIR, output_file)
            print(f"filename: {filename}")
        else:
            filename = os.path.join(output_file, "log.txt")
        if not os.path.exists(LOGS_DIR):
            os.makedirs(LOGS_DIR)
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
