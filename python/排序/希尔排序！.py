# -*- coding: utf-8 -*-
# @Time: 2020/5/7 13:26
import numpy as np

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]

def shellSort(arr):
    # 希尔排序
    gap = len(arr) // 2
    while gap > 0:
        for i in range(gap, len(arr)):
            j = i
            while (j >= gap) and (arr[j] < arr[j-gap]):
                swap(arr, j, j-gap)
                j = j-gap
        gap = gap//2
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