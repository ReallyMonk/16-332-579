#!/usr/bin/env python
# coding=utf-8
'''
@Author: Jin X
@Date: 2020-04-07 22:38:00
@LastEditTime: 2020-04-08 18:06:23
'''

import pickle
import numpy as np


def load():
    with open(
            "F:\Rutgers\\2ndSemester\Intro to DL\Homework\Project\\mnist.pkl",
            'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist[
        "test_images"], mnist["test_labels"]


x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)
x_train = x_train.astype(float)
x_test = x_test.astype(float)
x_train = x_train / 255.0 * 0.99 + 0.01
x_test = x_test / 255.0 * 0.99 + 0.01

step_size = 1e-1
reg = 1e-5
Num = len(y_train)


def relu(Z):
    return np.maximum(0, Z)


def relu_backward(dA, Z):
    dA[Z <= 0] = 0
    return dA


def softmax(x):
    row_max = x.max(axis=1, keepdims=True)
    normal_x = x - row_max
    normal_x = x - row_max
    exp_x = np.exp(normal_x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def corssEntropyLoss(out, labels, epoch):
    predict = softmax(out)
    if epoch % 10 == 0:
        accuracy = np.sum(predict.argmax(axis=1) == labels) / Num
        print('epoch: {},training accuracy: {:.2f}'.format(
            epoch, accuracy * 100))
        with open('F:\Rutgers\\2ndSemester\Intro to DL\Homework\Project\\train.txt', 'a') as f:
            f.write('{}:{}\n'.format(epoch, accuracy))
    predict[range(Num), labels] -= 0.99
    dL = predict
    return dL


def forward(x, w, b):
    return x.dot(w) + b


def backprop(dA, x, w, b, i):
    dw = x.T.dot(dA)
    db = np.sum(dA, axis=0, keepdims=True)
    dx = dA.dot(w[i].T)
    dw /= Num
    db /= Num
    w[i] -= step_size * dw
    b[i] -= step_size * db

    return dx


def testing(test, labels, w, b):
    TestNum = len(labels)
    h1 = relu(forward(test, w[0], b[0]))
    h2 = relu(forward(h1, w[1], b[1]))
    out = forward(h2, w[2], b[2])
    predict = softmax(out)
    accuracy = np.sum(predict.argmax(axis=1) == labels) / TestNum
    print('testing accuracy is: {:.2f}'.format(accuracy * 100))
    with open('F:\Rutgers\\2ndSemester\Intro to DL\Homework\Project\\test.txt', 'a') as f:
        f.write(str(accuracy)+'\n')


def training(train, labels, w, b, epochs=1000):

    h1, h2, out = None, None, None

    for i in range(epochs):
        h1 = relu(forward(train, w[0], b[0]))
        h2 = relu(forward(h1, w[1], b[1]))
        out = forward(h2, w[2], b[2])
        grad_out = corssEntropyLoss(out, labels, i)
        grad_h2 = relu_backward(grad_out, out)
        grad_h2 = backprop(grad_h2, h2, w, b, 2)
        grad_h1 = relu_backward(grad_h2, h2)
        grad_h1 = backprop(grad_h1, h1, w, b, 1)
        backprop(grad_h1, train, w, b, 0)
        if i % 10 == 0:
            testing(x_test, y_test, w, b)


def fcnn(train, trainLabels, test, testLabels):
    factor = 1e-1
    w = [
        np.random.randn(784, 200) * factor,
        np.random.randn(200, 50) * factor,
        np.random.randn(50, 10) * factor
    ]
    b = [
        np.random.randn(1, 200) * factor,
        np.random.randn(1, 50) * factor,
        np.random.randn(1, 10) * factor
    ]

    training(train, trainLabels, w, b)
    testing(test, testLabels, w, b)


fcnn(x_train, y_train, x_test, y_test)
