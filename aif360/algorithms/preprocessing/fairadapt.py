from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import rpy2

from aif360.algorithms import Transformer
from aif360.metrics import utils

from rpy2 import robjects
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
import rpy2.robjects.numpy2ri as numpy2ri


import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import has_fit_parameter

from aif360.sklearn.utils import check_inputs, check_groups


class fairadapt(BaseEstimator):
	"""
	Fairadapt is a pre-processing technique that can be used for fair classification
	or fair regression (Plecko & Meinshausen, JMLR 2020).
	References:
		JMLR paper: https://www.jmlr.org/papers/volume21/19-966/19-966.pdf [1]
		Github page: https://github.com/dplecko/fairadapt [2]
	"""
	def __init__(self,
				prot_attr,
				adj_mat,
				outcome
				):
		"""
		Args:
			:param prot_attr : Name of the protected attribute.
			:param adj_mat (array-like): A 2-dimensional array representing the adjacency matrix of the causal diagram of the data
			 	generating process.
			:param outcome: String containing the name of the desired response (dependent) variable.
		"""
		self.prot_attr = prot_attr
		self.adj_mat = adj_mat
		self.outcome = outcome

		# R packages need to run fairadapt
		packnames = ('ranger', 'fairadapt')
		# selectively install the missing packages
		names_to_install = [x for x in packnames if not robjects.packages.isinstalled(x)]
		if len(names_to_install) > 0:
			utils.install_packages(StrVector(names_to_install))


	def fit_transform(self, X_train, y_train, X_test):
		"""Train less biased classifier or regressor with the
		given training data.
		Args:
			X_train (pandas.DataFrame): Training data frame (including the protected attribute).
			y_train (array-like): Labels of the training samples.
			X_test (pandas.DataFrame): Test data frame (including the protected attribute).
		Returns:
			X_fair_train (pandas.DataFrame): Transformed training data.
			y_fair_train (array-like): Transformed labels of the training samples.
			X_fair_test (pandas.DataFrame): Transformed test data.

			, ,
		"""
		# merge X_train and y_train
		df_train = pd.concat([X_train, y_train], axis = 1)

		Fairadapt_R = robjects.r(
    		'''
		    # fairadapt called from R
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

		# run fairadapt in R
		res = Fairadapt_R(
    		train_data = df_train,
    		test_data = X_test,
    		adj_mat = self.adj_mat,
    		prot_attr = self.prot_attr,
    		outcome = self.outcome
		)

		train_adapt = res.rx2('train')
		train_adapt.columns = [self.outcome] + X_test.columns.tolist()
		y_fair_train = train_adapt[self.outcome]
		X_fair_train = train_adapt.drop([self.outcome], axis = 1)

		X_fair_test = res.rx2('test')
		X_fair_test.columns = X_test.columns

		return X_fair_train, y_fair_train, X_fair_test
