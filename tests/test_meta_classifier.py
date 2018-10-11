from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
sys.path.append("../")
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from aif360.algorithms.inprocessing.meta_fair_classifier import MetaFairClassifier
from aif360.algorithms.inprocessing.celisMeta.utils import getStats
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def test_adult():
	protected = 'sex'
	ad = AdultDataset(protected_attribute_names=[protected],
	    privileged_classes=[['Male']], categorical_features=[],
	    features_to_keep=['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'])

	#scaler = MinMaxScaler(copy=False)
	# ad.features = scaler.fit_transform(ad.features)

	train, test = ad.split([32562])
	assert np.any(test.labels)

	#print(test.labels)

	biased_model = MetaFairClassifier(tau=0, sensitive_attr=protected)
	biased_model.fit(train)

	dataset_bias_test = biased_model.predict(test)

	predictions = [1 if y == train.favorable_label else -1 for y in list(dataset_bias_test.labels)]
	y_test = np.array([1 if y == [train.favorable_label] else -1 for y in test.labels])
	x_control_test = pd.DataFrame(data=test.features, columns=test.feature_names)[protected]

	acc, sr, unconstrainedFDR = getStats(y_test, predictions, x_control_test)
	#print(unconstrainedFDR)


	tau = 0.9
	debiased_model = MetaFairClassifier(tau=tau, sensitive_attr=protected)
	debiased_model.fit(train)

	#dataset_debiasing_train = debiased_model.predict(dataset_orig_train)
	dataset_debiasing_test = debiased_model.predict(test)

	predictions = list(dataset_debiasing_test.labels)
	predictions = [1 if y == train.favorable_label else -1 for y in predictions]
	y_test = np.array([1 if y == [train.favorable_label] else -1 for y in test.labels])
	x_control_test = pd.DataFrame(data=test.features, columns=test.feature_names)[protected]

	acc, sr, fdr = getStats(y_test, predictions, x_control_test)
	#print(fdr, unconstrainedFDR)
	assert(fdr >= unconstrainedFDR)

#test_adult()