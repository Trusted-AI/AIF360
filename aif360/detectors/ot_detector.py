from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from typing import Union
import pandas as pd
import numpy as np
import ot

def _normalize(distribution1, distribution2):
    """
    Transform distributions to pleasure form, that is their sums are equal to 1,
    and in case if there is negative values, increase all values with absolute value of smallest number.

    Args:
        distribution1 (numpy array): nontreated distribution
        distribution2 (numpy array): nontreated distribution
    """
    if np.minimum(np.min(distribution1), np.min(distribution2)) < 0:
        extra = -np.minimum(np.min(distribution1), np.min(distribution2))
        distribution1 += extra
        distribution2 += extra
    
    total_of_distribution1 = np.sum(distribution1)
    distribution1 /= total_of_distribution1
    total_of_distribution2 = np.sum(distribution2)
    distribution2 /= total_of_distribution2

def _transform(golden_standard, classifier, data, cost_matrix=None):
    """
    Transoform given distributions from pandas type to numpy arrays, and _normalize them.
    Rearanges distributions, with totall data allocated of one.
    Generates matrix distance with respect to (golden_standard[i] - classifier[j])^2.

    Args:
        golden_standard (series): ground truth (correct) target values
        classifier (series,  dataframe, optional): pandas series estimated targets
            as returned by a model for binary, continuous and ordinal modes.
        data (dataframe): the dataset (containing the features) the model was trained on

    Returns:
        initial_distribution, which is an processed golden_standard (numpy array)
        required_distribution, which is an processed classifier (numpy array)
        matrix_distance, which stores the distances between the cells of distributions (2d numpy array)
    """
    initial_distribution = (pd.Series.to_numpy(golden_standard)).astype(float)
    required_distribution = (pd.Series.to_numpy(classifier)).astype(float)

    _normalize(initial_distribution, required_distribution)

    if cost_matrix is not None:
        matrix_distance = cost_matrix
    else:
        matrix_distance = np.array([abs(i - required_distribution) for i in initial_distribution], dtype=float)
    return initial_distribution, required_distribution, matrix_distance

def evaluate(golden_standard, classifier, data, num_iters, **kwargs):
    """If the given golden_standart and classifier are distributions, it returns the Wasserstein distance between them, 
    otherwise it extract all neccessary information from data, makes logistic regression and 
    compute optimal transport for all possible options for the given classifier.

    Args:
        golden_standard (series, str): ground truth (correct) target values
        classifier (series,  dataframe, str): pandas series estimated targets
        data (dataframe): the dataset (containing the features) the model was trained on
        num_iters (int, optional): number of iterations (random restarts). Should be positive.

    Returns:
        ot.emd2 (float, dict): Earth mover's distance or dictionary of optimal transports for each of option of classifier
    
    Raises:
        AssertionError: If golden_standart is presented as str and data under this column does not containe exactly two different values.
    """
    # If the input details are considered as string parameters, we should extract and make logic regression for golden_standart and classifier, 
    # otherwise calculate the optimal transport between already given distributions
    if isinstance(golden_standard, str) and isinstance(classifier, str):
        df = data.drop(golden_standard, axis = 1)
        parameters = df.columns
        vocabulary = {}
        for param in parameters:
            options = sorted(set(df[param]))
            counter = 0
            for opt in options:
                if param == classifier:
                    vocabulary[counter] = opt
                df[param][df[param] == opt] = counter # Changing our data which can be "str"-string parameter into "int"-integer
                counter += 1
        
        clf = LogisticRegression(solver='lbfgs', max_iter=10000, C=1.0, penalty='l2')

        assert len(set(data[golden_standard])) == 2, \
            f"The golden_standart attains only one possible value, but should contain exactly two."
        criterion = pd.Series(preprocessing.LabelEncoder().fit_transform(data[golden_standard])) # Changes values in golden_standart into "0" and "1", to be discrete
            
        clf.fit(df, criterion)
        arrangement = pd.Series(clf.predict_proba(df)[:,0]) 
        
        # Running optimal transport to all possible parameters for the corresponding input classifier
        output = {}
        options = sorted(set(df[classifier]))
        for opt in options:
             initial_distribution = data[df[classifier] == opt][golden_standard]
             required_distribution = arrangement[df[classifier] == opt]
             initial_distribution, required_distribution, matrix_distance = _transform(initial_distribution, required_distribution, data, kwargs.get("cost_matrix"))
             output[opt] = ot.emd2(a=initial_distribution, b=required_distribution, M=matrix_distance, numItermax=num_iters)

        # As data is stored in numbers, convert it back to a more presentable form
        result = {}
        for key in output:
            result[vocabulary[key]] = output[key]
        if len(result) == 1:
            return result[next(iter(result))]
        return dict(sorted(result.items()))
    else:
        initial_distribution, required_distribution, matrix_distance = _transform(golden_standard, classifier, data, kwargs.get("cost_matrix"))
        return ot.emd2(a=initial_distribution, b=required_distribution, M=matrix_distance, numItermax=num_iters)

def ot_bias_scan(
    golden_standard: Union[pd.Series, str],
    classifier: Union[pd.Series, pd.DataFrame, str],
    data: pd.DataFrame = None,
    favorable_value: Union[str, float] = None,
    overpredicted: bool = True,
    scoring: str = "Optimal Transport",
    num_iters: int = 100,
    penalty: float = 1e-17,
    mode: str = "ordinal",
    **kwargs,
):
    """Calculated the Wasserstein distance for two given distributions.
    Transforms pandas Series into numpy arrays, transofrms and normalize them.
    After all, solves the optimal transport problem.

    Args:
        golden_standard (series, str): ground truth (correct) target values
        classifier (series,  dataframe, str): pandas series estimated targets
            as returned by a model for binary, continuous and ordinal modes.
            If mode is nominal, this is a dataframe with columns containing classifier for each nominal class.
            If None, model is assumed to be a dumb model that predicts the mean of the targets
                    or 1/(num of categories) for nominal mode.
        data (dataframe): the dataset (containing the features) the model was trained on
        favorable_value(str, float, optional): Should be high or low or float if the mode in [binary, ordinal, or continuous].
                If float, value has to be minimum or maximum in the golden_standard column. Defaults to high if None for these modes.
                Support for float left in to keep the intuition clear in binary classification tasks.
                If mode is nominal, favorable values should be one of the unique categories in the golden_standard.
                Defaults to a one-vs-all scan if None for nominal mode.
        overpredicted (bool, optional): flag for group to scan for.
            True means we scan for a group whose classifier/predictions are systematically higher than observed.
            In other words, True means we scan for a group whose observeed is systematically lower than the classifier.
            False means we scan for a group whose classifier/predictions are systematically lower than observed.
            In other words, False means we scan for a group whose observed is systematically higher than the classifier.
        scoring (str or class): only 'Optimal Transport'
        num_iters (int, optional): number of iterations (random restarts). Should be positive.
        penalty (float, optional): penalty term. Should be positive. The penalty term as with any regularization parameter may need to be
            tuned for ones use case. The higher the penalty, the higher the influence of entropy regualizer.
        mode: one of ['binary', 'continuous', 'nominal', 'ordinal']. Defaults to binary.
                In nominal mode, up to 10 categories are supported by default.
                To increase this, pass in keyword argument max_nominal = integer value.

    Returns:
        ot.emd2 (float, dict): Earth mover's distance or dictionary of optimal transports for each of option of classifier

    Raises:
        AssertionError: If golden_standard is the type pandas.Series or str and classifier is the type pandas.Series or pandas.DataFrame or str.
        AssertionError: If cost_matrix is presented and its type is numpy.ndarray.
        AssertionError: If scoring variable is not "Optimal Transport".
        AssertionError: If type mode does not belong to any, of the possible options 
                        ["binary", "continuous", "nominal", "ordinal"].
        AssertionError: If golden distribution is presented as pandas.Series and favorable_value does not belong to any, of the possible options 
                        [min_val, max_val, "flag-all", *uniques].
    """
    # Inspect whether the types are correct for golden_standard and classifier
    assert isinstance(golden_standard, (pd.Series, str)) and isinstance(classifier, (pd.Series, pd.DataFrame, str)), \
        f"The type of golden_standard should be pandas.Series and classifier should be pandas.Series or pandas.DataFrame, but obtained {type(golden_standard)}, {type(classifier)}."
    
    if kwargs.get("cost_matrix") is not None:
        # Inspect whether the type is correct for cost_matrix
        assert isinstance(kwargs.get("cost_matrix"), np.ndarray), \
            f"The type of cost_matrix should be numpy.array, but obtained {type(kwargs.get('cost_matrix'))}"
    
    # Check whether scoring correspond to "Optimal Transport"
    assert scoring == "Optimal Transport", \
        f"Scoring mode can only be \"Optimal Transport\", got {scoring}."

    # Ensure correct mode is passed in.
    assert mode in ['binary', 'continuous', 'nominal', 'ordinal'], \
        f"Expected one of {['binary', 'continuous', 'nominal', 'ordinal']}, got {mode}."
    
    # Set classifier to mean targets for non-nominal modes
    if classifier is None and mode != "nominal":
        classifier = pd.Series(golden_standard.mean(), index=golden_standard.index)

    # Set correct favorable value (this tells us if higher or lower is better)
    if not isinstance(golden_standard, str):
        min_val, max_val = golden_standard.min(), golden_standard.max()
        uniques = list(golden_standard.unique())

        if favorable_value == 'high':
            favorable_value = max_val
        elif favorable_value == 'low':
            favorable_value = min_val
        elif favorable_value is None:
            if mode in ["binary", "ordinal", "continuous"]:
                favorable_value = max_val # Default to higher is better
            elif mode == "nominal":
                favorable_value = "flag-all" # Default to scan through all categories

        assert favorable_value in [min_val, max_val, "flag-all", *uniques,], \
            f"Favorable_value should be high, low, or one of categories {uniques}, got {favorable_value}."

    if mode == "binary": # Flip golden_standard if favorable_value is 0 in binary mode.
        golden_standard = pd.Series(golden_standard == favorable_value, dtype=int)
    elif mode == "nominal":
        unique_outs = set(sorted(golden_standard.unique()))
        size_unique_outs = len(unique_outs)
        if classifier is None: # Set classifier to 1/(num of categories) for nominal mode
            classifier = pd.Series(1 / golden_standard.nunique(), index=golden_standard.index)

        if favorable_value != "flag-all": # If favorable flag is set, use one-vs-others strategy to scan, else use one-vs-all strategy
            golden_standard = golden_standard.map({favorable_value: 1})
            golden_standard = golden_standard.fillna(0)
            if isinstance(classifier, pd.DataFrame):
                classifier = classifier[favorable_value]
        else:
            results = {}
            orig_golden_standard = golden_standard.copy()
            orig_classifier = classifier.copy()
            for unique in uniques:
                golden_standard = orig_golden_standard.map({unique: 1})
                golden_standard = golden_standard.fillna(0)

                if isinstance(classifier, pd.DataFrame):
                    classifier = orig_classifier[unique]

                result = evaluate(golden_standard, classifier, data, num_iters, **kwargs)
                results[unique] = result
            return results
    
    return evaluate(golden_standard, classifier, data, num_iters, **kwargs)
