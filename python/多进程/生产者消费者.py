# -*- coding: utf-8 -*-
# @Time: 2020/5/7 14:38
from multiprocessing import Process, Queue
import time

# 消费者方法
def consumer(q, name):
    while True:
        res = q.get()
        if res is None: break
        print("%s 吃了 %s" % (name, res))

# 生产者方法
def producer(q, name, food):
    for i in range(3):
        time.sleep(1)  # 模拟生产西瓜的时间延迟
        res = "%s %s" % (food, i)
        print("%s 生产了 %s" % (name, res))
        # 把生产的vegetable放入到队列中
        q.put(res)

if __name__ == "__main__":
    # 创建队列
    q = Queue()

    # 创建生产者
    p1 = Process(target=producer, args=(q, "kelly", "西瓜"))
    c1 = Process(target=consumer, args=(q, "peter",))

    p1.start()
    c1.start()

    p1.join()
    q.put(None)

