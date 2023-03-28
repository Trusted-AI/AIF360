import sys
sys.path.append("../../../script_fairXplainer_public/")
from fairxplainer.fair_explainer import FairXplainer
from fairxplainer.fair_explainer import plot as fif_plot
import numpy as np


def explain_statistical_parity(
        clf, 
        X, 
        sensitive_features, 
        maxorder=2, 
        spline_intervals=5,
        k=7,
        seed=22, 
        cpu_time=300, 
        verbose=False):
    """
        Explain the statistical parity of a classifier given a dataset

    Args:
        clf: the trained model to explain
            clf.predict() should be implemented
        X (pandas.DataFrame): the dataset (containing the features) to measure the bias on
        sensitive_features (list): list of sensitive attributes
        maxorder (int): maximum order of the intersectionality of features
        spline_intervals (int): number of intervals to use for the cubic spline approximation
        k (int): number of top feature interactions to show
        seed (int): seed for the random number generator
        cpu_time (int): maximum time to run the algorithm
        verbose (bool): whether to print the results

    Returns:
        pandas.DataFrame : the bias weights

    """

    fairXplainer = FairXplainer(
            clf, 
            X, 
            sensitive_features)
    
    fairXplainer.compute(
            maxorder=maxorder, 
            spline_intervals=spline_intervals,
            explain_sufficiency_fairness=False,
            seed=seed,
            cpu_time=cpu_time, 
            verbose=verbose)
    
    result = fairXplainer.get_top_k_weights(k=k)

    if(verbose):
        print("\nc Exact statistical parity", fairXplainer.statistical_parity_sample())

    return result, fairXplainer.statistical_parity_sample()

def explain_equalized_odds(
        clf,
        X,
        y_true,
        sensitive_features,
        maxorder=2,
        spline_intervals=5,
        k=7,
        seed=22,
        cpu_time=300,
        verbose=False):
    
    """
        Explain the equalized odds of a classifier given a dataset

    Args:
        clf: the trained model to explain
            clf.predict() should be implemented
        X (pandas.DataFrame): the dataset (containing the features) to measure the bias on
        y_true (pandas.Series): the labels of the dataset
        sensitive_features (list): list of sensitive attributes
        maxorder (int): maximum order of the intersectionality of features
        spline_intervals (int): number of intervals to use for the cubic spline approximation
        k (int): number of top feature interactions to show
        seed (int): seed for the random number generator
        cpu_time (int): maximum time to run the algorithm
        verbose (bool): whether to print the results

    Returns:
        pandas.DataFrame : the bias weights as list for y_true = 0 and y_true = 1

    """

    y_values = y_true.unique()    
    results = []
    bias_values = []
    for y_val in y_values:
        if(verbose):
            print("\nc Explaining for label", y_val)
        result, bias = explain_statistical_parity(
            clf,
            X[y_true == y_val],
            sensitive_features,
            maxorder=maxorder,
            spline_intervals=spline_intervals,
            k=k,
            seed=seed,
            cpu_time=cpu_time,
            verbose=verbose
        )

        results.append(result)
        bias_values.append(bias)

    return results, bias_values


def explain_predictive_parity(
        clf,
        X,
        y_true,
        sensitive_features,
        maxorder=2,
        spline_intervals=5,
        k=7,
        seed=22,
        cpu_time=300,
        verbose=False):
    """
        Explain the predictive parity of a classifier given a dataset

    Args:
        clf: the trained model to explain
            clf.predict() should be implemented
        X (pandas.DataFrame): the dataset (containing the features) to measure the bias on
        y_true (pandas.Series): the labels of the dataset
        sensitive_features (list): list of sensitive attributes
        maxorder (int): maximum order of the intersectionality of features
        spline_intervals (int): number of intervals to use for the cubic spline approximation
        k (int): number of top feature interactions to show
        seed (int): seed for the random number generator
        cpu_time (int): maximum time to run the algorithm
        verbose (bool): whether to print the results

    Returns:
        pandas.DataFrame : the bias weights as list for y_predicted = 0 and y_predicted = 1
    """

    y_predicted = clf.predict(X)
    y_values = np.unique(y_predicted)
    results = []
    bias_values = []
    for y_val in y_values:
        if(verbose):
            print("\nc Explaining for label", y_val)
        fairXplainer = FairXplainer(
            None, 
            X[y_predicted == y_val],
            sensitive_features,
            label=y_true[y_predicted == y_val]
        )
    
        fairXplainer.compute(
                maxorder=maxorder, 
                spline_intervals=spline_intervals,
                explain_sufficiency_fairness=True,
                seed=seed,
                compute_sp_only=False,
                cpu_time=cpu_time, 
                verbose=verbose)
        
        result = fairXplainer.get_top_k_weights(k=k)

        results.append(result)
        bias_values.append(fairXplainer.statistical_parity_sample())

    return results, bias_values


def draw_plot(result, **kwargs):
    """
        Draw the plot of the bias weights

    Args:
        result (pandas.DataFrame): the bias weights
        kwargs: arguments to pass to the plot function

    Returns:
        matplotlib.pyplot : the plot
    """

    plt = fif_plot(result, 
            **kwargs
    )
    plt.tight_layout()
    return plt

    