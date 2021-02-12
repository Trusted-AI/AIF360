"""
The code for GridSearchReduction wraps the source class 
fairlearn.reductions.GridSearch
available in the https://github.com/fairlearn/fairlearn library
licensed under the MIT Licencse, Copyright Microsoft Corporation
"""

from sklearn.base import BaseEstimator, ClassifierMixin
import fairlearn.reductions as red 
import numpy as np 
from sklearn.preprocessing import LabelEncoder

class GridSearchReduction(BaseEstimator, ClassifierMixin):
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
				prot_attr_cols,
				estimator, 
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
		self.prot_attr_cols = prot_attr_cols
		self.moments = {
						"DemographicParity": red.DemographicParity,
						"EqualizedOdds": red.EqualizedOdds,
						"TruePositiveRateDifference": red.TruePositiveRateDifference,
						"ErrorRateRatio": red.ErrorRateRatio,
						"GroupLoss": red.GroupLossMoment
						}
		
		if constraints is not None:
			try:
				self.moment = object.__new__(self.moments[constraints])
				if constraints is "GroupLoss":
					
					losses = {
							"ZeroOne":red.ZeroOneLoss,
							"Square": red.SquareLoss,
							"Absolute":red.AbsoluteLoss
							}
					
					self.loss = object.__new__(losses[loss])
					if loss is "ZeroOne":
						self.loss.__init__()
					else:
						self.loss.__init__(min_val, max_val)
					
					self.moment.__init__(loss=self.loss)
				
				else:
					self.moment.__init__()
			except KeyError as error:
				print("Invalid value for constraints: %s" % (error))
		else:
			if constraints_moment is None:
				raise ValueError('Parameter constraints or constraints_moment must be defined')
			self.moment = constraints_moment

		self.estimator = estimator
		self.constraint_weight = constraint_weight
		self.grid_size = grid_size
		self.grid_limit = grid_limit
		self.grid = grid 
		self.drop_prot_attr = drop_prot_attr

		self.model = red.GridSearch(estimator=self.estimator, constraints=self.moment, 
									constraint_weight=self.constraint_weight, grid_size=self.grid_size,
									grid_limit=self.grid_limit, grid=self.grid)



	def fit(self, X, y):
		"""Train less biased classifier or regressor with the
		given training data.
		Args:
			X (pandas.DataFrame): Training samples.
			y (array-like): Training output.
		Returns:
			self
		"""
		A = X[self.prot_attr_cols]

		if self.drop_prot_attr:
			X = X.drop(self.prot_attr_cols, axis=1)

		self.model.fit(X, y, sensitive_features=A)

		return self
	

	def predict(self, X):
		"""Predict output for the given samples.
		Args:
			X (pandas.DataFrame): Test samples.
		Returns:
			numpy.ndarray: Predicted output per sample.
		"""
		if self.drop_prot_attr:
			X = X.drop(self.prot_attr_cols, axis=1)
		
		return self.model.predict(X)


	def predict_proba(self, X):
		"""Probability estimates.
		The returned estimates for all classes are ordered by the label of
		classes for classification.
		Args:
			X (pandas.DataFrame): Test samples.
		Returns:
			numpy.ndarray: returns the probability of the sample for each class
			in the model, where classes are ordered as they are in
			``self.classes_``.
		"""
		if self.drop_prot_attr:
			X = X.drop(self.prot_attr_cols)
		
		if isinstance(self.model.constraints, red.ClassificationMoment):
			return self.model.predict_proba(X)

		raise NotImplementedError("Underlying model does not support predict_proba")
