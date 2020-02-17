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
mini_test = np.random.randint(20, size=20)
mini_test = mini_test.reshape(10, 2)


# Define knn classifier
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
            clfr_label.append(mini_train_label[idx])

        # dicide wich class the input data should be
        label_counts = Counter(clfr_label)
        test_class = label_counts.most_common(1)

        result.append(test_class[0][0])
        print(test_class[0][0])
        print(clfr_label)

    return result


outputlabels = kNNClassify(mini_test, mini_train, mini_train_label, 10)

print('random test points are:', mini_test)
print('knn classfied labels for test:', outputlabels)

# plot train data and classfied test data
train_x = mini_train[:, 0]
train_y = mini_train[:, 1]
fig = plt.figure()
plt.scatter(train_x[np.where(mini_train_label == 0)],
            train_y[np.where(mini_train_label == 0)],
            color='red')
plt.scatter(train_x[np.where(mini_train_label == 1)],
            train_y[np.where(mini_train_label == 1)],
            color='blue')
plt.scatter(train_x[np.where(mini_train_label == 2)],
            train_y[np.where(mini_train_label == 2)],
            color='green')
plt.scatter(train_x[np.where(mini_train_label == 3)],
            train_y[np.where(mini_train_label == 3)],
            color='black')

test_x = mini_test[:, 0]
test_y = mini_test[:, 1]
outputlabels = np.array(outputlabels)
plt.scatter(test_x[np.where(outputlabels == 0)],
            test_y[np.where(outputlabels == 0)],
            marker='*',
            color='red')
plt.scatter(test_x[np.where(outputlabels == 1)],
            test_y[np.where(outputlabels == 1)],
            marker='*',
            color='blue')
plt.scatter(test_x[np.where(outputlabels == 2)],
            test_y[np.where(outputlabels == 2)],
            marker='*',
            color='green')
plt.scatter(test_x[np.where(outputlabels == 3)],
            test_y[np.where(outputlabels == 3)],
            marker='*',
            color='black')

# save diagram as png file
plt.savefig("D:\Rutgers/2nd Semester\Intro to DL\Homework\Pic\miniknn.png")
