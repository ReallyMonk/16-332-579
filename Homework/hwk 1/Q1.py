import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import heapq
from collections import Counter
mpl.use('Agg')

# load mini training data and labels
mini_train = np.load(
    'D:\Rutgers/2nd Semester\Intro to DL\Homework\knn_minitrain.npy')
mini_train_label = np.load(
    'D:\Rutgers/2nd Semester\Intro to DL\Homework\knn_minitrain_label.npy')

# randomly generate test data

nums = [1, 1, 1, 1, 8, 2, 23, 7, -4, 18, 23, 24, 37, 2]

# 最大的3个数的索引
max_num_index_list = map(nums.index, heapq.nlargest(3, nums))

# 最小的3个数的索引
min_num_index_list = map(nums.index, heapq.nsmallest(3, nums))

'''
print(list(max_num_index_list))
print(list(min_num_index_list))
'''

k = range(4)
print(k[3])