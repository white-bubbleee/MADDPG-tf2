# -*- coding: utf-8 -*-
# @Date       : 2024/5/2417:31
# @Auther     : Wang.zr
# @File name  : distribution.py
# @Description:
import tensorflow as tf


@tf.function
def gen_action_for_discrete(actions):
    u = tf.random.uniform(tf.shape(actions), dtype=tf.float64)
    return tf.nn.softmax(actions - tf.math.log(-tf.math.log(u)), axis=-1)


@tf.function
def gen_action_for_continuous(actions):
    mean, logstd = tf.split(axis=1, num_or_size_splits=2, value=actions)
    std = tf.exp(logstd)
    return mean + std * tf.random.normal(tf.shape(mean))
