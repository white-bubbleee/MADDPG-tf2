# -*- coding: utf-8 -*-
# @Date       : 2024/5/2422:48
# @Auther     : Wang.zr
# @File name  : logger.py
# @Description:

import logging
import os
from datetime import datetime

LOGS_EXP_DIR = None

def set_logger(name=None, output_file=None, log_dir=None, console=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s %(levelname)s] %(module)s:%(funcName)s:%(lineno)s: %(message)s')

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    global LOGS_EXP_DIR
    if log_dir is None and LOGS_EXP_DIR is None:
        curtime = datetime.now()
        cur_dir = f"{curtime.strftime('%Y-%m-%d-%H-%M-%S')}"
        log_dir = "../logs/" + cur_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        LOGS_EXP_DIR = log_dir
    elif log_dir is None and LOGS_EXP_DIR is not None:
        log_dir = LOGS_EXP_DIR


    if output_file is not None:
        if output_file.endswith(".txt") or output_file.endswith(".log"):
            filename = os.path.join(log_dir, output_file)
            print(f"logger saved: {filename}")
        else:
            filename = os.path.join(output_file, "log.txt")
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
