# Based on code from https://github.com/zjelveh/learning-fair-representations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax


def LFR_optim_objective(parameters, x_unpriveleged, x_priveleged, y_unpriveleged,
                        y_priveleged, k=10, A_x=0.01, A_y=0.1, A_z=0.5, print_inteval=250):

    num_unpriveleged, features_dim = x_unpriveleged.shape
    num_priveleged, _ = x_priveleged.shape

    w = parameters[:k]
    prototypes = parameters[k:].reshape((k, features_dim))

    M_unpriveleged, x_hat_unpriveleged, y_hat_unpriveleged = get_xhat_y_hat(prototypes, w, x_unpriveleged)

    M_priveleged, x_hat_priveleged, y_hat_priveleged = get_xhat_y_hat(prototypes, w, x_priveleged)

    y_hat = np.concatenate([y_hat_unpriveleged, y_hat_priveleged], axis=0)
    y = np.concatenate([y_unpriveleged.reshape((-1, 1)), y_priveleged.reshape((-1, 1))], axis=0)

    L_x = np.mean((x_hat_unpriveleged - x_unpriveleged) ** 2) + np.mean((x_hat_priveleged - x_priveleged) ** 2)
    L_z = np.mean(abs(np.mean(M_unpriveleged, axis=0) - np.mean(M_priveleged, axis=0)))
    L_y = - np.mean(y * np.log(y_hat) + (1. - y) * np.log(1. - y_hat))

    total_loss = A_x * L_x + A_y * L_y + A_z * L_z

    if LFR_optim_objective.steps % print_inteval == 0:
        print("step: {}, loss: {}, L_x: {},  L_y: {},  L_z: {}".format(
            LFR_optim_objective.steps, total_loss, L_x,  L_y,  L_z))
    LFR_optim_objective.steps += 1

    return total_loss


def get_xhat_y_hat(prototypes, w, x):
    M = softmax(cdist(x, prototypes), axis=1)
    x_hat = np.matmul(M, prototypes)
    y_hat = np.clip(
        np.matmul(M, w.reshape((-1, 1))),
        1e-6,
        0.999
    )
    return M, x_hat, y_hat
