from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pandas as pd
import numpy as np

from aif360.metrics import utils
from sklearn.base import BaseEstimator

from rpy2 import robjects
from rpy2.robjects.vectors import StrVector


class FairAdapt(BaseEstimator):
    """Fair Data Adaptation.

    Fairadapt is a pre-processing technique that can be used for both fair
    classification and fair regression [#plecko20]_. The method is a causal
    inference approach to bias removal and it relies on the causal graph for
    the dataset. The original implementation is in R [#plecko21]_.

    References:
    .. [#plecko20] `D. Plečko and N. Meinshausen,
       "Fair Data Adaptation with Quantile Preservation,"
       Journal of Machine Learning Research, 2020
       <https://www.jmlr.org/papers/volume21/19-966/19-966.pdf>`_
    .. [#plecko21] `D. Plečko and N. Bennett and N. Meinshausen,
       "FairAdapt: Causal Reasoning for Fair Data Pre-processing," arXiv 2021
       <https://arxiv.org/abs/2110.10200>`_

    Attributes:
        prot_attr_ (str or list(str)): Protected attribute(s) used for
            fair data adaptation.
        groups_ (array, shape (n_groups,)): A list of group labels known to the
            transformer.
    """

    def __init__(self,
                 prot_attr,
                 adj_mat,
                 outcome
                 ):
        """
        Args:
            :param prot_attr : Name of the protected attribute.
            :param adj_mat (array-like): A 2-dimensional array representing the adjacency matrix of the causal diagram
                of the data generating process.
            :param outcome: String containing the name of the desired response (dependent) variable.

        """
        self.prot_attr = prot_attr
        self.prot_attr_ = prot_attr
        self.adj_mat = adj_mat
        self.outcome = outcome

        # R packages need to run FairAdapt
        packnames = ('ranger', 'fairadapt')
        # selectively install the missing packages
        names_to_install = [x for x in packnames if not robjects.packages.isinstalled(x)]
        if len(names_to_install) > 0:
            utls = robjects.packages.importr('utils')
            utls.chooseCRANmirror(ind=1)
            utls.install_packages(StrVector(names_to_install))

    def fit_transform(self, X_train, y_train, X_test):
        """Remove bias from the given dataset by fair adaptation.

        Args:
            X_train (pandas.DataFrame): Training data frame (including the protected attribute).
            y_train (array-like): Training labels.
            X_train (pandas.DataFrame): Test data frame (including the protected attribute).

        Returns:
            X_fair_train (pandas.DataFrame): Transformed training data frame.
            y_fair_train (array-like): Transformed training labels.
            X_fair_train (pandas.DataFrame): Transformed test data frame.

        """

        # merge X_train and y_train
        df_train = pd.concat([X_train, y_train], axis=1)
        self.groups_ = np.unique(X_train[self.prot_attr])

        Fairadapt_R = robjects.r(
            '''
		    # FairAdapt called from R
		    function(train_data, test_data, adj_mat, res_vars = NULL, prot_attr,
					 outcome) {

				prot_attr <- gsub("-", ".", prot_attr)
				outcome <- gsub("-", ".", outcome)

		        train_data <- as.data.frame(
		            lapply(train_data, function(x) {
		                if (is.ordered(x)) class(x) <- "factor"
		                x
		            })
		        )

		        test_data <- as.data.frame(
		            lapply(test_data, function(x) {
		                if (is.ordered(x)) class(x) <- "factor"
		                x
		            })
		        )

		        adj.mat <- as.matrix(adj_mat)
		        rownames(adj.mat) <- colnames(adj.mat) <- names(train_data)

		        formula_adult <- as.formula(paste(outcome, "~ ."))
		        L <- fairadapt::fairadapt(
		            formula = formula_adult,
		            train.data = train_data,
		            test.data = test_data,
		            adj.mat = adj.mat,
		            prot.attr = prot_attr,
		            res.vars = res_vars
		        )

		        names(L) <- c("train", "test")
		        return(L)
		    }
		    '''
        )

        # run FairAdapt in R
        res = Fairadapt_R(
            train_data=df_train,
            test_data=X_test,
            adj_mat=self.adj_mat,
            prot_attr=self.prot_attr,
            outcome=self.outcome
        )

        train_adapt = res.rx2('train')
        train_adapt.columns = [self.outcome] + X_test.columns.tolist()
        y_fair_train = train_adapt[self.outcome]
        X_fair_train = train_adapt.drop([self.outcome], axis=1)

        X_fair_test = res.rx2('test')
        X_fair_test.columns = X_test.columns

        return X_fair_train, y_fair_train, X_fair_test
