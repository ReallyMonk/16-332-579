import numpy as np
from download_mnist import load
# import operator
import time
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

print(x_train.shape)
print(x_test.shape)