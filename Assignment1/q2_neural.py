# -*-coding:utf-8-*-
import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    实现单隐层神经网络前后向传播
    """
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # 前向传播
    h = sigmoid(np.dot(X, W1) + b1)
    yhat = softmax(np.dot(h, W2) + b2)

    # 后向传播
    cost = np.sum(-np.log(yhat[labels == 1])) / X.shape[0]
    d3 = (yhat - labels) / X.shape[0]
    gradW2 = np.dot(h.T, d3)
    gradb2 = np.sum(d3, 0, keepdims=True)
    dh = np.dot(d3, W2.T)
    grad_h = sigmoid_grad(h) * dh
    gradW1 = np.dot(X.T, grad_h)
    gradb1 = np.sum(grad_h, 0)

    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
                           gradW2.flatten(), gradb2.flatten()))
    return cost, grad


def sanity_check():
    """
    网络可用性测试
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0, dimensions[2] - 1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:forward_backward_prop(data, labels, params, dimensions), params)


if __name__ == "__main__":
    sanity_check()
