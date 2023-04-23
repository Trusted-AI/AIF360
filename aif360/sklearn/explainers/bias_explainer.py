from fairxplainer.fair_explainer import FairXplainer
from fairxplainer.fair_explainer import plot as fif_plot
import numpy as np


def fairxplain_statistical_parity(
        clf, 
        X, 
        prot_attr, 
        maxorder=2, 
        spline_intervals=5,
        k=7,
        seed=None, 
        cpu_time=300, 
        verbose=False):
    """
        Explain the statistical parity of a classifier given a dataset

    Args:
        clf: the trained model to explain
        clf.predict() should be implemented
        X (pandas.DataFrame): the dataset (containing the features) to measure the bias on
        prot_attr (list): list of protected or sensitive attributes
        maxorder (int): maximum order of the intersectionality of features
        spline_intervals (int): number of intervals to use for the cubic spline approximation
        k (int): number of top feature interactions to show
        seed (int or None): seed for the random number generator
        cpu_time (int): maximum time to run the algorithm
        verbose (bool): whether to print the results

    Returns:
        pandas.DataFrame : the bias weights

    """

    fairXplainer = FairXplainer(
            clf, 
            X, 
            prot_attr)
    
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

def fairxplain_equalized_odds(
        clf,
        X,
        y_true,
        prot_attr,
        maxorder=2,
        spline_intervals=5,
        k=7,
        seed=None,
        cpu_time=300,
        verbose=False):
    
    """
        Explain the equalized odds of a classifier given a dataset

    Args:
        clf: the trained model to explain
        clf.predict() should be implemented
        X (pandas.DataFrame): the dataset (containing the features) to measure the bias on
        y_true (pandas.Series): the labels of the dataset
        prot_attr (list): list of protected or sensitive attributes
        maxorder (int): maximum order of the intersectionality of features
        spline_intervals (int): number of intervals to use for the cubic spline approximation
        k (int): number of top feature interactions to show
        seed (int or None): seed for the random number generator
        cpu_time (int): maximum time to run the algorithm
        verbose (bool): whether to print the results

    Returns:
        list of pandas.DataFrame : the bias weights as list for y_true = 0 and y_true = 1

    """

    y_values = y_true.unique()    
    results = []
    bias_values = []
    for y_val in y_values:
        if(verbose):
            print("\nc Explaining for label", y_val)
        result, bias = fairxplain_statistical_parity(
            clf,
            X[y_true == y_val],
            prot_attr,
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


def fairxplain_predictive_parity(
        clf,
        X,
        y_true,
        prot_attr,
        maxorder=2,
        spline_intervals=5,
        k=7,
        seed=None,
        cpu_time=300,
        verbose=False):
    """
        Explain the predictive parity of a classifier given a dataset

    Args:
        clf: the trained model to explain
        clf.predict() should be implemented
        X (pandas.DataFrame): the dataset (containing the features) to measure the bias on
        y_true (pandas.Series): the labels of the dataset
        prot_attr (list): list of protected or sensitive attributes
        maxorder (int): maximum order of the intersectionality of features
        spline_intervals (int): number of intervals to use for the cubic spline approximation
        k (int): number of top feature interactions to show
        seed (int or None): seed for the random number generator
        cpu_time (int): maximum time to run the algorithm
        verbose (bool): whether to print the results

    Returns:
        list of pandas.DataFrame : the bias weights as list for y_predicted = 0 and y_predicted = 1
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
            prot_attr,
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


def draw_plot(
        result,
        draw_waterfall=True, 
        fontsize=22, 
        labelsize=18, 
        figure_size=(10, 5), 
        title="", 
        xlim=None,
        x_label="Influence", 
        text_x_pad=0.02, 
        text_y_pad=0.1, 
        result_x_pad=0.02, 
        result_y_location=0.5, 
        delete_zero_weights=False
    ):
    """
        Draw the plot of the bias weights

    Args:
        result (pandas.DataFrame): the bias weights
        draw_waterfall (bool): whether to draw the waterfall plot
        fontsize (int): the fontsize of the title
        labelsize (int): the fontsize of the labels
        figure_size (tuple): the size of the figure
        title (str): the title of the plot
        xlim (tuple): the limits of the x-axis
        x_label (str): the label of the x-axis
        text_x_pad (float): the padding of the text on the bar along the x-axis
        text_y_pad (float): the padding of the text on the bar along the y-axis
        result_x_pad (float): the padding of the metric value along the x-axis
        result_y_location (float): the location of the metric value along the y-axis
        delete_zero_weights (bool): whether to delete the zero weights

    Returns:
        matplotlib.pyplot : the plot
    """

    plt = fif_plot(result, 
        draw_waterfall=draw_waterfall,
        fontsize=fontsize,
        labelsize=labelsize,
        figure_size=figure_size,
        title=title,
        xlim=xlim,
        x_label=x_label,
        text_x_pad=text_x_pad,
        text_y_pad=text_y_pad,
        result_x_pad=result_x_pad,
        result_y_location=result_y_location,
        delete_zero_weights=delete_zero_weights
    )
    plt.tight_layout()
    return plt

    