"""
The code for GridSearchReduction wraps the source class 
fairlearn.reductions.GridSearch
available in the https://github.com/fairlearn/fairlearn library
licensed under the MIT Licencse, Copyright Microsoft Corporation
"""

from aif360.algorithms import Transformer
from aif360.sklearn.inprocessing import GridSearchReduction as skGridSearchRed
import numpy as np 
import pandas as pd 
import fairlearn.reductions as red 


class GridSearchReduction(Transformer): 
    """ 
    Grid search is an in-processing technique that can be used for fair classification
    or fair regression. For classification it reduces fair classification to a sequence of 
    cost-sensitive classification problems, returning the deterministic classifier 
    with the lowest empirical error subject to fair classification constraints [1]_ among
    the candidates searched. For regression it uses the same priniciple to return a 
    deterministic regressor with the lowest empirical error subject to the constraint of
    bounded group loss [2]_.
    References:
        .. [1] A. Agarwal, A. Beygelzimer, M. Dudik, J. Langford, and H. Wallach,
            "A Reductions Approach to Fair Classification," International Conference 
            on Machine Learning, 2018.
        .. [2] A. Agarwal, M. Dudik, and Z. Wu, "Fair Regression: Quantitative Definitions 
            and Reduction-based Algorithms," International Conference on Machine Learning,
            2019.
    """
    def __init__(self,
                 estimator, 
                 prot_attr_cols=None,
                 constraints=None,
                 constraints_moment=None, 
                 constraint_weight=0.5,
                 grid_size=10,
                 grid_limit=2.0,
                 grid=None,
                 drop_prot_attr = True,
                 loss="ZeroOne",
                 min_val = None,
                 max_val = None
                 ):
        """
        Args:
            :param prot_attr_cols: String or array-like column indices or column names of protected attributes
            :param estimator: An function implementing methods :code:`fit(X, y, sample_weight)` and
                :code:`predict(X)`, where `X` is the matrix of features, `y` is the vector of labels, and
                `sample_weight` is a vector of weights i.e sklearn classifiers/regressors
            :param constraints: Optional string keyword denoting the `fairlearn.reductions.moment` object 
                defining the disparity constraints i.e "DemographicParity" or "EqualizedOdds". For a full 
                list of possible options see `ExponentiatedGradientReduction.moments`. If None, 
                parameter `constraints_moment` must be defined.
            :param constraints_moment: `fairlearn.reductions.moment` object defining the disparity
                constraints. If None, pararmeter `constraints` must be defined. 
            :param constraint_weight: When the `selection_rule` is "tradeoff_optimization" (default, no other option 
                currently) this float specifies the relative weight put on the constraint violation when selecting the 
                best model. The weight placed on the error rate will be :code:`1-constraint_weight`
            :param grid_size: The number of Lagrange multipliers to generate in the grid (int)
            :param grid_limit: The largest Lagrange multiplier to generate. The grid will contain values
                distributed between :code:`-grid_limit` and :code:`grid_limit` by default (float)
            :param grid: Instead of supplying a size and limit for the grid, users may specify the exact
                set of Lagrange multipliers they desire using this argument in a pandas dataframe
            :param drop_prot_attr: Boolean flag indicating whether to drop protected attributes from 
                training data
            :param loss: String identifying loss function for constraints. Options include "ZeroOne", "Square", 
                and "Absolute."
            :param min_val: Loss function parameter for "Square" and "Absolute," typically the minimum of the range
                of y values
            :param max_val: Loss function parameter for "Square" and "Absolute," typically the maximum of the range
                of y values
        """
        super(GridSearchReduction, self).__init__()

        #init model, set prot_attr_cols during fit
        if prot_attr_cols is None:
            prot_attr_cols =[]
        self.model = skGridSearchRed(prot_attr_cols, estimator, constraints, constraints_moment, 
                 constraint_weight, grid_size, grid_limit, grid, drop_prot_attr, loss, min_val, max_val)


    def fit(self, dataset):
        """Learns model with less bias
        
        Args:
            dataset : Dataset containing true output.
        
        Returns:
            GridSearchReduction: Returns self.
        """
        #set prot_attr_cols
        if len(self.model.prot_attr_cols)==0:
            self.model.prot_attr_cols = dataset.protected_attribute_names

        X_df = pd.DataFrame(dataset.features, columns = dataset.feature_names) 
        Y = dataset.labels
        
        self.model.fit(X_df, Y)

        return self
    

    def predict(self, dataset):
        """Obtain the predictions for the provided dataset using the 
         model learned.
        Args:
            dataset : Dataset containing output values that needs
                to be transformed.
        Returns:
            dataset : Transformed dataset.
        """
        X_df = pd.DataFrame(dataset.features, columns = dataset.feature_names) 

        dataset_new = dataset.copy(deepcopy = True)
        dataset_new.labels = self.model.predict(X_df).reshape(-1,1) 

        if isinstance(self.model.constraints, red.ClassificationMoment):
            try:
                #Probability of favorable label
                scores = self.model.predict_proba(X_df)[:,int(dataset.favorable_label)]
                dataset_new.scores = scores.reshape(-1, 1)
            except: 
                print("dataset.scores not updated, underlying model does not support predict_proba")
            
        return dataset_new
      










