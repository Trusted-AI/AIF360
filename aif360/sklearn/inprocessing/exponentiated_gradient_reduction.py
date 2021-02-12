"""
The code for ExponentiatedGradientReduction wraps the source class 
fairlearn.reductions.ExponentiatedGradient
available in the https://github.com/fairlearn/fairlearn library
licensed under the MIT Licencse, Copyright Microsoft Corporation
"""

from sklearn.base import BaseEstimator, ClassifierMixin
import fairlearn.reductions as red 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from aif360.sklearn.utils import check_inputs

class ExponentiatedGradientReduction(BaseEstimator, ClassifierMixin):
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
                 prot_attr_cols,
                 estimator, 
                 constraints=None,
                 constraints_moment=None, 
                 eps=0.01, 
                 T=50, 
                 nu=None, 
                 eta_mul=2.0,
                 drop_prot_attr = True 
                 ):
        """
        Args:
            :param prot_attr_cols: String or array-like column indices or column names of protected attributes
            :param estimator: A function implementing methods :code:`fit(X, y, sample_weight)` and
                :code:`predict(X)`, where `X` is the matrix of features, `y` is the vector of labels, and
                `sample_weight` is a vector of weights; labels `y` and predictions returned by 
                :code:`predict(X)` are either 0 or 1 i.e sklearn classifiers
            :param constraints: Optional string keyword denoting the `fairlearn.reductions.moment` object 
                defining the disparity constraints i.e "DemographicParity" or "EqualizedOdds". For a full 
                list of possible options see `ExponentiatedGradientReduction.moments`. If None, 
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
        self.prot_attr_cols = prot_attr_cols
        self.moments = {
                        "DemographicParity": red.DemographicParity,
                        "EqualizedOdds": red.EqualizedOdds,
                        "TruePositiveRateDifference": red.TruePositiveRateDifference,
                        "ErrorRateRatio": red.ErrorRateRatio
                        }
        
        if constraints is not None:
            try:
                self.moment = object.__new__(self.moments[constraints])
                self.moment.__init__()
                
            except KeyError as error:
                print("Invalid value for constraints: %s" % (error))
        else:
            if constraints_moment is None:
                raise ValueError('Parameter constraints or constraints_moment must be defined')
            self.moment = constraints_moment

        self.estimator = estimator
        self.eps=eps 
        self.T=T 
        self.nu=nu
        self.eta_mul=eta_mul
        self.drop_prot_attr = drop_prot_attr

        self.model = red.ExponentiatedGradient(self.estimator, self.moment, self.eps, self.T, self.nu, self.eta_mul)



    def fit(self, X, y):
        """Learns randomized model with less bias
        
        Args:
            X (pandas.DataFrame): Training samples.
            y (array-like): Training labels.
        Returns:
            self
        """
        A = X[self.prot_attr_cols]

        if self.drop_prot_attr:
            X = X.drop(self.prot_attr_cols, axis=1)

        le = LabelEncoder()
        y = le.fit_transform(y)
        self.classes_ = le.classes_

        self.model.fit(X, y, sensitive_features=A)

        return self
    

    def predict(self, X):
        """Predict class labels for the given samples.
        Args:
            X (pandas.DataFrame): Test samples.
        Returns:
            numpy.ndarray: Predicted class label per sample.
        """
        if self.drop_prot_attr:
            X = X.drop(self.prot_attr_cols, axis=1)
        
        return self.classes_[self.model.predict(X)]


    def predict_proba(self, X):
        """Probability estimates.
        The returned estimates for all classes are ordered by the label of
        classes.
        Args:
            X (pandas.DataFrame): Test samples.
        Returns:
            numpy.ndarray: Returns the probability of the sample for each class
            in the model, where classes are ordered as they are in
            ``self.classes_``.
        """
        if self.drop_prot_attr:
            X = X.drop(self.prot_attr_cols)
        
        return self.model._pmf_predict(X)


