import numpy as np
import matplotlib.pyplot as plt

# 目标函数:y=x^2
def func(x):
    return np.square(x)

# 目标函数一阶导数:dy/dx=2*x
def dfunc(x):
    return 2 * x

def GD_momentum(x_start, df, epochs, lr, momentum):
    xs = np.zeros(epochs+1)
    x = x_start
    xs[0] = x
    v = 0
    for i in range(epochs):
        dx = df(x)
        # v表示x要改变的幅度
        v = - dx * lr + momentum * v
        x += v
        xs[i+1] = x
    return xs


def demo2_GD_momentum():
    line_x = np.linspace(-5, 5, 100)
    line_y = func(line_x)
    plt.figure('Gradient Desent: Learning Rate, Momentum')

    x_start = -5
    epochs = 6

    lr = [0.01, 0.1, 0.6, 0.9]
    momentum = [0.0, 0.1, 0.5, 0.9]

    color = ['k', 'r', 'g', 'y']

    plt.figure(figsize=(14, 10))
    row = len(lr)
    col = len(momentum)
    size = np.ones(epochs + 1) * 10
    size[-1] = 70
    for i in range(row):
        for j in range(col):
            x = GD_momentum(x_start, dfunc, epochs, lr=lr[i], momentum=momentum[j])
            plt.subplot(row, col, i * col + j + 1)
            plt.plot(line_x, line_y, c='b')
            plt.plot(x, func(x), c=color[i], label='lr={}, mo={}'.format(lr[i], momentum[j]))
            plt.scatter(x, func(x), c=color[i], s=size)
            plt.legend(loc=0)
    plt.show()


demo2_GD_momentum()
