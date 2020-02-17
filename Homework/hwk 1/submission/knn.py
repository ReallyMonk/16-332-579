# import math
import numpy as np
from download_mnist import load
# import operator
import time
import heapq
from collections import Counter

# classify using kNN
# x_train = np.load('../x_train.npy')
# y_train = np.load('../y_train.npy')
# x_test = np.load('../x_test.npy')
# y_test = np.load('../y_test.npy')
x_train_ori, y_train, x_test_ori, y_test = load()
x_train_ori = x_train_ori.reshape(60000, 28, 28)
x_test_ori = x_test_ori.reshape(10000, 28, 28)
x_train_ori = x_train_ori.astype(float)
x_test_ori = x_test_ori.astype(float)

# turn each entry into an 1-D array
x_train = []
x_test = []
for X_pic in x_train_ori:
    tmp = []
    for X in X_pic:
        tmp = np.append(tmp, X)
    x_train.append(tmp)

for X_pic in x_test_ori:
    tmp = []
    for X in X_pic:
        tmp = np.append(tmp, X)
    x_test.append(tmp)

x_train = np.array(x_train)
x_test = np.array(x_test)


def kNNClassify(newInput, dataSet, labels, k):
    result = []

    for test_In in newInput:

        # record the k nearest neibour for each new point
        clfr_dist = []
        clfr_label = []
        for train_En in dataSet:
            clfr_dist.append(np.linalg.norm(train_En - test_In))
        k_min_index = map(clfr_dist.index, heapq.nsmallest(k, clfr_dist))

        # find the labels of those distance refer to
        for idx in k_min_index:
            clfr_label.append(y_train[idx])

        # dicide wich class the input data should be
        label_counts = Counter(clfr_label)
        test_class = label_counts.most_common(1)

        result.append(test_class[0][0])
        print(test_class[0][0])
        print(clfr_label)

    return result


start_time = time.time()
outputlabels = kNNClassify(x_test[600:630], x_train, y_train, 10)
result = y_test[600:630] - outputlabels
result = (1 - np.count_nonzero(result) / len(outputlabels))
print("---classification accuracy for knn on mnist: %s ---" % result)
print("---execution time: %s seconds ---" % (time.time() - start_time))
