"""This is the helper script for implementing metrics."""
import numpy as np


def compute_boolean_conditioning_vector(X, feature_names, condition=None):
    """Compute the boolean conditioning vector.

    Args:
        X (numpy.ndarray): Dataset features
        feature_names (list): Names of the features.
        condition (list(dict)): Specifies the subset of instances we want to
            use. Format is a list of `dicts` where the keys are `feature_names`
            and the values are values in `X`. Elements in the list are clauses
            joined with OR operators while key-value pairs in each dict are
            joined with AND operators. See examples for more details. If `None`,
            the condition specifies the entire set of instances, `X`.

    Returns:
        numpy.ndarray(bool): Boolean conditioning vector. Shape is `[n]` where
        `n` is `X.shape[0]`. Values are `True` if the corresponding row
        satisfies the `condition` and `False` otherwise.

    Examples:
        >>> condition = [{'sex': 1, 'age': 1}, {'sex': 0}]

        This corresponds to `(sex == 1 AND age == 1) OR (sex == 0)`.
    """
    if condition is None:
        return np.ones(X.shape[0], dtype=bool)

    overall_cond = np.zeros(X.shape[0], dtype=bool)
    for group in condition:
        group_cond = np.ones(X.shape[0], dtype=bool)
        for name, val in group.items():
            index = feature_names.index(name)
            group_cond = np.logical_and(group_cond, X[:, index] == val)
        overall_cond = np.logical_or(overall_cond, group_cond)

    return overall_cond

def compute_num_instances(X, w, feature_names, condition=None):
    """Compute the number of instances, :math:`n`, conditioned on the protected
    attribute(s).

    Args:
        X (numpy.ndarray): Dataset features.
        w (numpy.ndarray): Instance weight vector.
        feature_names (list): Names of the features.
        condition (list(dict)): Same format as
            :func:`compute_boolean_conditioning_vector`.

    Returns:
        int: Number of instances (optionally conditioned).
    """

    # condition if necessary
    cond_vec = compute_boolean_conditioning_vector(X, feature_names, condition)

    return np.sum(w[cond_vec], dtype=np.float64)

def compute_num_pos_neg(X, y, w, feature_names, label, condition=None):
    """Compute the number of positives, :math:`P`, or negatives, :math:`N`,
    optionally conditioned on protected attributes.

    Args:
        X (numpy.ndarray): Dataset features.
        y (numpy.ndarray): Label vector.
        w (numpy.ndarray): Instance weight vector.
        feature_names (list): Names of the features.
        label (float): Value of label (unfavorable/positive or
            unfavorable/negative).
        condition (list(dict)): Same format as
            :func:`compute_boolean_conditioning_vector`.

    Returns:
        int: Number of positives/negatives (optionally conditioned)
    """
    y = y.ravel()
    cond_vec = compute_boolean_conditioning_vector(X, feature_names,
        condition=condition)
    return np.sum(w[np.logical_and(y == label, cond_vec)], dtype=np.float64)

def compute_num_TF_PN(X, y_true, y_pred, w, feature_names, favorable_label,
                      unfavorable_label, condition=None):
    """Compute the number of true/false positives/negatives optionally
    conditioned on protected attributes.

    Args:
        X (numpy.ndarray): Dataset features.
        y_true (numpy.ndarray): True label vector.
        y_pred (numpy.ndarray): Predicted label vector.
        w (numpy.ndarray): Instance weight vector - the true and predicted
            datasets are supposed to have same instance level weights.
        feature_names (list): names of the features.
        favorable_label (float): Value of favorable/positive label.
        unfavorable_label (float): Value of unfavorable/negative label.
        condition (list(dict)): Same format as
            :func:`compute_boolean_conditioning_vector`.

    Returns:
        Number of positives/negatives (optionally conditioned).
    """
    # condition if necessary
    cond_vec = compute_boolean_conditioning_vector(X, feature_names,
        condition=condition)

    # to prevent broadcasts
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    y_true_pos = (y_true == favorable_label)
    y_true_neg = (y_true == unfavorable_label)
    y_pred_pos = np.logical_and(y_pred == favorable_label, cond_vec)
    y_pred_neg = np.logical_and(y_pred == unfavorable_label, cond_vec)

    # True/false positives/negatives
    return dict(
        TP=np.sum(w[np.logical_and(y_true_pos, y_pred_pos)], dtype=np.float64),
        FP=np.sum(w[np.logical_and(y_true_neg, y_pred_pos)], dtype=np.float64),
        TN=np.sum(w[np.logical_and(y_true_neg, y_pred_neg)], dtype=np.float64),
        FN=np.sum(w[np.logical_and(y_true_pos, y_pred_neg)], dtype=np.float64)
    )

def compute_num_gen_TF_PN(X, y_true, y_score, w, feature_names, favorable_label,
                    unfavorable_label, condition=None):
    """Compute the number of generalized true/false positives/negatives
    optionally conditioned on protected attributes. Generalized counts are based
    on scores and not on the hard predictions.

    Args:
        X (numpy.ndarray): Dataset features.
        y_true (numpy.ndarray): True label vector.
        y_score (numpy.ndarray): Predicted score vector. Values range from 0 to
            1. 0 implies prediction for unfavorable label and 1 implies
            prediction for favorable label.
        w (numpy.ndarray): Instance weight vector - the true and predicted
            datasets are supposed to have same instance level weights.
        feature_names (list): names of the features.
        favorable_label (float): Value of favorable/positive label.
        unfavorable_label (float): Value of unfavorable/negative label.
        condition (list(dict)): Same format as
            :func:`compute_boolean_conditioning_vector`.

    Returns:
        Number of positives/negatives (optionally conditioned).
    """
    # condition if necessary
    cond_vec = compute_boolean_conditioning_vector(X, feature_names,
        condition=condition)

    # to prevent broadcasts
    y_true = y_true.ravel()
    y_score = y_score.ravel()
    w = w.ravel()

    y_true_pos = np.logical_and(y_true == favorable_label, cond_vec)
    y_true_neg = np.logical_and(y_true == unfavorable_label, cond_vec)

    # Generalized true/false positives/negatives
    return dict(
        GTP=np.sum((w*y_score)[y_true_pos], dtype=np.float64),
        GFP=np.sum((w*y_score)[y_true_neg], dtype=np.float64),
        GTN=np.sum((w*(1.0-y_score))[y_true_neg], dtype=np.float64),
        GFN=np.sum((w*(1.0-y_score))[y_true_pos], dtype=np.float64)
    )

def compute_distance(X_orig, X_distort, X_prot, feature_names, dist_fun,
                     condition=None):
    """Compute the distance element-wise for two sets of vectors.

    Args:
        X_orig (numpy.ndarray): Original features.
        X_distort (numpy.ndarray): Distorted features. Shape must match
            `X_orig`.
        X_prot (numpy.ndarray): Protected attributes (used to compute
            condition). Should be same for both original and distorted.
        feature_names (list): Names of the protected features.
        dist_fun (function): Function which returns the distance (float) between
            two 1-D arrays (e.g. :func:`scipy.spatial.distance.euclidean`).
        condition (list(dict)): Same format as
            :func:`compute_boolean_conditioning_vector`.

    Returns:
        (numpy.ndarray(numpy.float64), numpy.ndarray(bool)):

            * Element-wise distances (1-D).
            * Condition vector (1-D).
    """
    cond_vec = compute_boolean_conditioning_vector(X_prot, feature_names,
        condition=condition)

    num_instances = X_orig[cond_vec].shape[0]
    distance = np.zeros(num_instances, dtype=np.float64)
    for i in range(num_instances):
        distance[i] = dist_fun(X_orig[cond_vec][i], X_distort[cond_vec][i])

    return distance, cond_vec
