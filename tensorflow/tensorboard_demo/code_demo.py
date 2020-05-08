# -*- coding: utf-8 -*-
# @Time: 2020/5/8 15:51
import warnings
warnings.filterwarnings(action='ignore')

# 重点：
#     1、tf.namescope()、 tf.summary.FileWriter('logs/',ss.graph)
#     2、参数名、op都可以定义新的名称，如下A、B、C
#     3、启动tensorboard：tensorboard --logdir=C:\Users\vip_g\logs
#     4、用Google浏览器打开生成的http地址
#

import tensorflow as tf

#定义命名空间
with tf.name_scope('input'):
  #fetch：就是同时运行多个op的意思
  input1 = tf.constant(3.0,name='A')#定义名称，会在tensorboard中代替显示
  input2 = tf.constant(4.0,name='B')
  input3 = tf.constant(5.0,name='C')
with tf.name_scope('op'):
  #加法
  add = tf.add(input2,input3)
  #乘法
  mul = tf.multiply(input1,add)
with tf.Session() as ss:
  #默认在当前py目录下的logs文件夹，没有会自己创建
  wirter = tf.summary.FileWriter('logs/',ss.graph)
  result = ss.run([mul,add])
  print(result)


'''
cmd:
    tensorboard --logdir=G:\03.git_self\learning_note\tensorflow\tensorboard_demo\logs
    注意：当执行指令报错时，尝试调整tensorboard的版本，当前版本：
        tensorflow == 1.13.1
        tensorboard == 1.11
'''