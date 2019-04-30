import numpy as np


def softmax(x):
    """
    一个二维矩阵或者向量的softmax结果
    :param x:
    :return:
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # 此时必然是一个矩阵
        exp_minmax = lambda x: np.exp(x - np.max(x))  # 利用exp映射为正数（减去最大值是为了压缩映射结果不至于过大导致溢出）
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax, 1, x)
        denominator = np.apply_along_axis(denom, 1, x)

        if len(denominator.shape) == 1:
            # 行向量调整为列向量
            denominator = denominator.reshape((denominator.shape[0], 1))

        x = x * denominator
    else:
        # 此时为向量
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator = 1.0 / np.sum(numerator)
        x = numerator.dot(denominator)

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    几个测试
    """
    print("Running basic tests...")

    test1 = softmax(np.array([1, 2]))
    print(test1)
    ans1 = np.array([0.26894142, 0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    print(test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001, -1002]]))
    print(test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)


if __name__ == "__main__":
    test_softmax_basic()