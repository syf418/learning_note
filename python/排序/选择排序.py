# -*- coding: utf-8 -*-
# @Time: 2020/5/7 13:29
import numpy as np

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]

def selectSort(arr):
    '''
    选择排序是，遍历一遍，找到最小的元素，然后交换到最左边，重复这个过程。和冒泡法相比，它遍历一遍，只交换一次。
    :param arr:
    :return:
    '''
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        swap(arr, min_idx, i)
        print("{} -> {}".format(i, arr))
    return arr

def get_arr(num=10):
    # 生成随机的大小为num的数组
    np.random.seed()
    arr = np.random.randint(0, num, (num,))
    return arr


if __name__ == "__main__":
    # 选择排序
    arr = get_arr()
    print('arr_ori:', arr)
    arr = selectSort(arr)
    print("arr_sorted:", arr)