import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np

def plot_single(errors_t, fp_diff_t, max_iters, gamma, C):
    # plot errors
    x = range(max_iters - 1)
    y_t = errors_t
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(x, y_t)
    plt.ylabel('average error of mixture')
    plt.xlabel('iterations')
    plt.title('error vs. time: C: {}, gamma: {}'.format(C, gamma))
    plt.show()

    # plot fp disparity
    x = range(max_iters - 1)
    y_t = fp_diff_t
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(x, y_t)
    plt.ylabel('fp_diff*group_size')
    plt.xlabel('iterations')
    plt.title('fp_diff*size vs. time: C: {}, gamma: {}'.format(C, gamma))
    ax2.plot(x, [gamma] * len(y_t))
    plt.show()