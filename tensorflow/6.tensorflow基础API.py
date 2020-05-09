# -*- coding: utf-8 -*-
'''
@Time    : 2020/1/2 14:38
@Author  : shangyf
@File    : 6.tensorflow基础API.py
'''
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

import platform
print(platform.python_version())

import tensorflow as tf
tf.enable_eager_execution()

print(tf.__version__)

# 常量
# t = tf.constant([[1,2,3], [4,5,6]])
t1 = tf.constant([5])
t2 = tf.constant([2])
t3 = t1 + t2
print("t3:", t3)

