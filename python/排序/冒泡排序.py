# -*- coding: utf-8 -*-
# @Time: 2020/5/7 13:10
import numpy as np

def bubbleSort(arr):
    # 冒泡法:
    '''
    从左到右遍历，遍历的元素和后一个比较，如果前一个比后一个大，则交换；第一次遍历后，最大的元素在最右的位置。
    以相同的方式遍历，次大的元素放置在倒数第2的位置；直到需要比较的元素只有1个为止。
    :param arr:
    :return:
    '''
    for i in range(len(arr) - 1, 0, -1):
        for j in range(i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
        print("{} -> {}".format(i, arr))
    return arr


def get_arr(num=10):
    # 生成随机的大小为num的数组
    np.random.seed()
    arr = np.random.randint(0, num, (num,))
    return arr


if __name__ == "__main__":
    # 冒泡法
    arr = get_arr()
    print('arr_ori:', arr)
    arr = bubbleSort(arr)
    print("arr_sorted:", arr)
