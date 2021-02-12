"""
The code for ExponentiatedGradientReduction wraps the source class 
fairlearn.reductions.ExponentiatedGradient
available in the https://github.com/fairlearn/fairlearn library
licensed under the MIT Licencse, Copyright Microsoft Corporation
"""

from aif360.algorithms import Transformer
from aif360.sklearn.inprocessing import ExponentiatedGradientReduction as skExpGradRed
import numpy as np 
import pandas as pd 

class ExponentiatedGradientReduction(Transformer):
    """ 
    Exponentiated gradient reduction is an in-processing technique that reduces 
    fair classification to a sequence of cost-sensitive classification problems, 
    returning a randomized classifier with the lowest empirical error subject to 
    fair classification constraints [1]_.  
    References:
        .. [1] A. Agarwal, A. Beygelzimer, M. Dudik, J. Langford, and H. Wallach,
            "A Reductions Approach to Fair Classification," International Conference 
            on Machine Learning, 2018.
    """
    def __init__(self,
                 estimator, 
                 constraints=None,
                 constraints_moment=None, 
                 eps=0.01, 
                 T=50, 
                 nu=None, 
                 eta_mul=2.0,
                 drop_prot_attr = True):
        """
        Args:
            :param estimator: A function implementing methods :code:`fit(X, y, sample_weight)` and
                :code:`predict(X)`, where `X` is the matrix of features, `y` is the vector of labels, and
                `sample_weight` is a vector of weights; labels `y` and predictions returned by 
                :code:`predict(X)` are either 0 or 1 i.e sklearn classifiers
            :param constraints: Optional string keyword denoting the `fairlearn.reductions.moment` object 
                defining the disparity constraints i.e "DemographicParity" or "EqualizedOdds". For a full 
                list of possible options see `self.model.moments`. If None, 
                parameter `constraints_moment` must be defined.
            :param constraints_moment: `fairlearn.reductions.moment` object defining the disparity 
                constraints. If None, pararmeter `constraints` must be defined.
            :param eps: Allowed fairness constraint violation; the solution is guaranteed to have the
                error within :code:`2*best_gap` of the best error under constraint eps; the constraint
                violation is at most :code:`2*(eps+best_gap).`
            :param T: Maximum number of iterations
            :param nu: Convergence threshold for the duality gap, corresponding to a
                conservative automatic setting based on the statistical uncertainty in measuring
                classification error
            :param eta_mul: Initial setting of the learning rate
            :param drop_prot_attr: Boolean flag indicating whether to drop protected attributes from 
                training data
            
        """
        super(ExponentiatedGradientReduction, self).__init__()

        #init model, set prot_attr_cols during fit
        prot_attr_cols =[]
        self.model = skExpGradRed(prot_attr_cols=prot_attr_cols, estimator=estimator, constraints=constraints,
            constraints_moment=constraints_moment, eps=eps, T=T, nu=nu, eta_mul=eta_mul, drop_prot_attr=drop_prot_attr)


    def fit(self, dataset):
        """Learns randomized model with less bias
        
        Args:
            dataset : (Binary label) Dataset containing true labels.
        
        Returns:
            ExponentiatedGradientReduction: Returns self.
        """
        #set prot_attr_cols
        self.model.prot_attr_cols = dataset.protected_attribute_names

        X_df = pd.DataFrame(dataset.features, columns = dataset.feature_names) 
        Y = dataset.labels
        
        self.model.fit(X_df, Y)

        return self
    

    def predict(self, dataset):
        """Obtain the predictions for the provided dataset using the 
        randomized model learned.
        Args:
            dataset : (Binary label) Dataset containing labels that needs
                to be transformed.
        Returns:
            dataset : Transformed (Binary label) dataset.
        """
        X_df = pd.DataFrame(dataset.features, columns = dataset.feature_names) 

        dataset_new = dataset.copy(deepcopy = True)
        dataset_new.labels = self.model.predict(X_df).reshape(-1,1) 

        
        try:
            #Probability of favorable label
            scores = self.model.predict_proba(X_df)[:,int(dataset.favorable_label)]
            dataset_new.scores = scores.reshape(-1, 1)
        except: 
            print("dataset.scores not updated, underlying model does not support predict_proba")
        
        return dataset_new
      










