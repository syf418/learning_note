# -*- coding: utf-8 -*-
# @Time: 2020/5/7 13:35
import numpy as np

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]

def heapSort(arr):
    '''
    堆排序使用二叉树的数据结构，需要满足，父节点值不小于儿子节点。
    堆排序，通过建立一个最大堆，即根节点的值最大,然后把根节点和最后一个叶节点交换；
    剩下的数重新建立一个最大堆，获得次大值，继续和叶节点交换；重复这个过程，直到堆的大小为1。
    :param arr:
    :return:
    '''
    def siftDown(arr, index, length=None):
        '''sift_down'''
        if length is None:
            length = len(arr)
        # 最大堆；子节点总小于父节点, O(n)
        while True:
            left = 2*index + 1
            right = 2*index + 2
            max_idx = left
            if left >= length:
                break
            if right<length and arr[right] > arr[left]:
                max_idx = right
            if arr[index] < arr[max_idx]:
                arr[index], arr[max_idx] = arr[max_idx],arr[index]
                index = max_idx
            else:
                break
    # 初始化最大堆 O(n)
    for idx in range(len(arr)//2-1, -1, -1):
        siftDown(arr, idx, len(arr))

    # 堆排序，交换最大值和最后一个叶节点, O(nlogn)
    for length in range(len(arr)-1, -1, -1):
        arr[length], arr[0] = arr[0], arr[length]
        siftDown(arr, 0, length)
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