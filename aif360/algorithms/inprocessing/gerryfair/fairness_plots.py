# Copyright 2019 Seth V. Neel, Michael J. Kearns, Aaron L. Roth, Zhiwei Steven Wu
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from matplotlib import pyplot as plt


def plot_single(errors_t, fp_diff_t, max_iters, gamma, C):
    """Plot the errors and false positive rate disparity over time.

    :param errors_t: list of errors at each iteration
    :param fp_diff_t: list of fp rate disparity at each iteration
    :param max_iters: time horizon T of the algorithm
    :param gamma: input gamma disparity
    :param C: input C parameter - see gerryfair_classifier.py
    """
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