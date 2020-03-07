from math import sin, cos, exp
import numpy as np
from numpy import matmul


def derivative(x, w):
    # computational graph
    p = x[0] * w[0]
    q = x[1] * w[1]
    l = sin(p)
    m = l * l
    n = cos(q)
    s = 2 + m + n
    f = 1 / s

    # derivation by computational graph
    df = -1 / (s * s)
    ds = 1
    dm = 2 * l
    dn = -sin(q)
    dl = cos(p)
    dp_x = w[0]
    dp_w = x[0]
    dq_x = w[1]
    dq_w = x[1]

    df_x1 = df * ds * dm * dl * dp_x
    df_w1 = df * ds * dm * dl * dp_w
    df_x2 = df * ds * dn * dq_x
    df_w2 = df * ds * dn * dq_w

    # derivation by hand
    dx1 = -2 * w[0] * sin(x[0] * w[0]) * cos(
        x[0] * w[0]) / (2 + sin(w[0] * x[0])**2 + cos(w[1] * x[1]))**2
    dw1 = -2 * x[0] * sin(x[0] * w[0]) * cos(
        x[0] * w[0]) / (2 + sin(w[0] * x[0])**2 + cos(w[1] * x[1]))**2
    dx2 = w[1] * sin(
        w[1] * x[1]) / (2 + sin(w[0] * x[0])**2 + cos(w[1] * x[1]))**2
    dw2 = x[1] * sin(
        w[1] * x[1]) / (2 + sin(w[0] * x[0])**2 + cos(w[1] * x[1]))**2

    print('df_x1 by hand', dx1)
    print('df_x1 by program', df_x1)
    print('df_x2 by hand', dx2)
    print('df_x2 by program', df_x2)
    print('df_w1 by hand', dw1)
    print('df_w1 by program', df_w1)
    print('df_w2 by hand', dw2)
    print('df_w2 by program', df_w2)


# derivative([1, 1], [2, 2])


def sig(x):
    res = []
    for i in range(0, 3):
        #print(x[i])
        tmp = 1 / (1 + exp(-x[i]))
        #print(tmp)
        res.append(tmp)
    res = np.array(res)
    res = res.reshape(3, 1)
    return res


def div_sig(x):
    res = []
    for i in range(0, 3):
        #print(x[i])
        tmp = 1 / (1 + exp(-x[i]))
        tp = (1 - tmp) * tmp
        #print(tp)
        res.append(tp)
    res = np.array(res)
    res = res.reshape(3, 1)
    return res


def mat_deriv(x, w):
    # computational graph
    p = matmul(w, x)
    q = sig(p)
    f = np.linalg.norm(q)**2

    # derivation
    df = 2 * q
    print('df', df)
    #print(df.shape)
    dq = div_sig(p)
    print('dq', dq)
    dp_x = w.T
    #print(dp_x.shape)
    dp_w = x.reshape(3, 1)
    #print(dp_w.shape)
    print('df*dq', df * dq)

    df_x = matmul(dp_x, df * dq)
    df_w = np.outer(dp_w.T, df * dq)
    print('df_w')
    print(df_w)
    print('df_x')
    print(df_x)


x = np.array([1, 0.2, 0.5])
w = np.array([1, 1, 1, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2])
w = w.reshape(3, 3)
print(x)
print(w)
mat_deriv(x, w)
