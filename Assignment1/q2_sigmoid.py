# -*-coding:utf-8-*-
# !/usr/bin/env python

import numpy as np


def sigmoid(x):
    """
    Python实现sigmoid函数
    """
    s = 1.0 / (1 + np.exp(-x))
    return s


def sigmoid_grad(s):
    """
    计算sigmoid梯度
    """
    ds = s * (1 - s)
    return ds


def test_sigmoid_basic():
    """
    几个测试
    """
    print("Running basic tests...")
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print(f)
    f_ans = np.array([
        [0.73105858, 0.88079708],
        [0.26894142, 0.11920292]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print(g)
    g_ans = np.array([
        [0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)


if __name__ == "__main__":
    test_sigmoid_basic()
