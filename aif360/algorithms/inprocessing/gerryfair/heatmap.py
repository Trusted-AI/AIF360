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
"""Function generating 3-d heatmap visualizing gamma-disparity.

The main function in this module, heat_map(), generates and saves a 3-d
heatmap visualizing the gamma-disparity for groups defined by linear thresholds over 2 sensitive attributes.
This serves as a (heuristic) method to help visualize convergence of the algorithm via brute force checking in
low dimensions, rather than relying on the Auditor. See [KRNW18] for details.
"""

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from aif360.algorithms.inprocessing.gerryfair.reg_oracle_class import *


def calc_disp(predictions, X, group_labels, X_prime, group):
    """Return the fp disparity in a group g - helper function for heat_map.

    Args:
        :param predictions: dataframe of predictions of the classifier
        :param X: dataframe of covariates
        :param group_labels: dataframe of group labels
        :param X_prime: sensitive covariates
        :param group: object of class Group(), see auditor.py
    Returns:
        :return: weighted disparity on the group g
    """
    X_0 = pd.DataFrame(
        [X_prime.iloc[u, :] for u, s in enumerate(group_labels) if s == 0])
    group_0 = group.predict(X_0)
    n = len(group_labels)
    g_size_0 = np.sum(group_0) * 1.0 / n
    FP = [predictions[i] for i, c in enumerate(group_labels) if c == 0]
    FP = np.mean(FP)
    group_members = group.predict(X_prime)
    fp_g = [
        predictions[i] for i, c in enumerate(group_labels)
        if group_members[i] == 1 and c == 0
    ]
    if len(fp_g) == 0:
        return 0
    fp_g = np.mean(fp_g)
    return (FP - fp_g) * g_size_0


def heat_map(X, X_prime, y, predictions, eta, plot_path, vmin=None, vmax=None):
    """Generate 3-d heatmap and save it at plot_path.
    Args:
        :param eta: discretization parameter of coefficients defining subgroups
        :param plot_path: the path to save the heatmap at
        :param vmin: Min  value to map: see plot_surface documentation in matplotlib
        :param vmax: Max value to map

    Returns:
        :return: the min and max gamma disparities on groups in the plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    columns = [str(c) for c in X_prime.columns]
    attribute_1 = np.zeros(int(1 / eta))
    attribute_2 = np.zeros(int(1 / eta))
    disparity = np.zeros((int(1 / eta), int(1 / eta)))

    for i in range(int(1 / eta)):
        for j in range(int(1 / eta)):
            beta = [-1 + 2 * eta * i, -1 + 2 * eta * j]
            group = LinearThresh(beta)

            attribute_1[i] = beta[0]
            attribute_2[j] = beta[1]
            disparity[i, j] = calc_disp(predictions, X, y, X_prime, group)

    X_plot, Y_plot = np.meshgrid(attribute_1, attribute_2)

    ax.set_xlabel(columns[0] + ' coefficient')
    ax.set_ylabel(columns[1] + ' coefficient')
    ax.set_zlabel('gamma disparity')
    ax.set_zlim3d([np.min(disparity), np.max(disparity)])
    surface = ax.plot_surface(X_plot,
                              Y_plot,
                              disparity,
                              cmap=cm.coolwarm,
                              linewidth=0,
                              antialiased=False,
                              vmin=vmin,
                              vmax=vmax)
    if plot_path != '.':
        fig.savefig('{}.png'.format(plot_path))
        plt.close()
    return [np.min(disparity), np.max(disparity)]
