# -*- coding: utf-8 -*-
# @Time: 2020/5/7 13:15
import numpy as np

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]

def insertionSort(arr):
    '''
    插入排序，是把当前的数插入到前面排序好的序列相应位置，使得序列保持单调性。
    :param arr:
    :return:
    '''
    for i in range(len(arr)):
        for j in range(i, 0, -1):
            if arr[j] < arr[j-1]:
                swap(arr, j, j-1)
            else:
                break
        print("{} -> {}".format(i, arr))
    return arr

def get_arr(num=10):
    # 生成随机的大小为num的数组
    np.random.seed()
    arr = np.random.randint(0, num, (num,))
    return arr

if __name__ == "__main__":
    arr = get_arr()
    print('arr_ori:', arr)
    arr = insertionSort(arr)
    print("arr_sorted:", arr)
