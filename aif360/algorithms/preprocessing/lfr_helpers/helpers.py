# Based on code from https://github.com/zjelveh/learning-fair-representations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax


def LFR_optim_objective(parameters, x_unprivileged, x_privileged, y_unprivileged,
                        y_privileged, k=10, A_x=0.01, A_y=0.1, A_z=0.5, print_interval=250, verbose=1):

    num_unprivileged, features_dim = x_unprivileged.shape
    num_privileged, _ = x_privileged.shape

    w = parameters[:k]
    prototypes = parameters[k:].reshape((k, features_dim))

    M_unprivileged, x_hat_unprivileged, y_hat_unprivileged = get_xhat_y_hat(prototypes, w, x_unprivileged)

    M_privileged, x_hat_privileged, y_hat_privileged = get_xhat_y_hat(prototypes, w, x_privileged)

    y_hat = np.concatenate([y_hat_unprivileged, y_hat_privileged], axis=0)
    y = np.concatenate([y_unprivileged.reshape((-1, 1)), y_privileged.reshape((-1, 1))], axis=0)

    L_x = np.mean((x_hat_unprivileged - x_unprivileged) ** 2) + np.mean((x_hat_privileged - x_privileged) ** 2)
    L_z = np.mean(abs(np.mean(M_unprivileged, axis=0) - np.mean(M_privileged, axis=0)))
    L_y = - np.mean(y * np.log(y_hat) + (1. - y) * np.log(1. - y_hat))

    total_loss = A_x * L_x + A_y * L_y + A_z * L_z

    if verbose and LFR_optim_objective.steps % print_interval == 0:
        print("step: {}, loss: {}, L_x: {},  L_y: {},  L_z: {}".format(
            LFR_optim_objective.steps, total_loss, L_x,  L_y,  L_z))
    LFR_optim_objective.steps += 1

    return total_loss


def get_xhat_y_hat(prototypes, w, x):
    M = softmax(-cdist(x, prototypes), axis=1)
    x_hat = np.matmul(M, prototypes)
    y_hat = np.clip(
        np.matmul(M, w.reshape((-1, 1))),
        np.finfo(float).eps,
        1.0 - np.finfo(float).eps
    )
    return M, x_hat, y_hat
