# -*- coding: utf-8 -*-
# @Date       : 2024/5/2218:54
# @Auther     : Wang.zr
# @File name  : utils.py
# @Description:
# -*- coding: utf-8 -*-
# @Date       : 2024/5/2013:32
# @Auther     : Wang.zr
# @File name  : utils.py
# @Description:
import os
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt


def save_model(arglist, exp_time, times, chechpoint):
    save_exp_dir = arglist.save_dir + arglist.exp_name + '/' + exp_time + '/'
    if not os.path.exists(save_exp_dir):
        os.makedirs(save_exp_dir)
    save_cur_dir = save_exp_dir + str(times) + '/'
    if not os.path.exists(save_cur_dir):
        os.makedirs(save_cur_dir)
    chechpoint.save(save_cur_dir)
    print(f"--Save to {save_cur_dir}")


def load_model(path, chechpoint):
    # 加载模型
    chechpoint.restore(tf.train.latest_checkpoint(path)).expect_partial()


def load_data2_plot(data_path, name: str, show=False):
    # 从.pkl文件中加载列表
    with open(data_path, 'rb') as f:
        data_list = pickle.load(f)

    # 画图
    plt.plot(data_list)
    plt.title(name)
    # plt.show()
    file_dir = os.path.dirname(data_path) + '/'
    plt.savefig(file_dir + name + '.png')
    plt.close()

    if show:
        plt.plot(data_list)
        plt.title(name)
        plt.show()


def save_config_2yaml(arglist, exp_time):
    # 保存配置文件
    save_exp_dir = arglist.save_dir + arglist.exp_name + '/' + exp_time + '/'
    if not os.path.exists(save_exp_dir):
        os.makedirs(save_exp_dir)
    with open(save_exp_dir + 'config.yaml', 'w') as f:
        for key, value in arglist.__dict__.items():
            f.write(f'{key}: {value}\n')