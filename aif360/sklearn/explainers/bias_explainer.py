from fairxplainer.fair_explainer import FairXplainer
from fairxplainer.fair_explainer import plot as fif_plot


def explain_statistical_parity(
        clf, 
        X, 
        sensitive_features, 
        maxorder=2, 
        spline_intervals=6,
        draw_plot=False,
        seed=22, 
        cpu_time=300, 
        verbose=False):
    """
        Explain the bias of a model given a dataset

    Args:
        clf: the trained model to explain
            clf.predict() should be implemented
        X (pandas.DataFrame): the dataset (containing the features) to measure the bias on
        sensitive_features (list): list of sensitive attributes
        maxorder (int): maximum order of the intersectionality of features
        spline_intervals (int): number of intervals to use for the cubic spline approximation
        draw_plot (bool): whether to draw the plot
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
    
    result = fairXplainer.get_top_k_weights()

    if draw_plot:
        plt = fif_plot(result, 
               draw_waterfall=True, 
               figure_size=(6.5,4), 
        )
        plt.tight_layout()
        plt.show()

    return result

    