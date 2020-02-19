from aif360.algorithms.inprocessing.gerryfair_classifier import *
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import *
import pytest
from aif360.metrics.binary_label_dataset_metric import *

@pytest.fixture
def gerry_model():
    max_iterations = 100
    C = 100
    print_flag = True
    gamma = .005

    gerry_model = GerryFairClassifier(C=C, printflag=print_flag, gamma=gamma, fairness_def='FP',
                                     max_iters=max_iterations, heatmapflag=False)
    return gerry_model


@pytest.fixture
def adult_dataset():
    adult_dataset = load_preproc_data_adult(sub_samp=1000, balance=True)
    return adult_dataset

# test train, predict, audit
def test_auditor(adult_dataset, gerry_model):
    gerry_model.fit(adult_dataset, early_termination=True)
    dataset_predicted = gerry_model.predict(adult_dataset, threshold=False)
    gerry_metric = BinaryLabelDatasetMetric(adult_dataset)
    gamma_disparity = gerry_metric.rich_subgroup(dataset_predicted.labels, 'FP')
    assert gamma_disparity >= 0
    assert gamma_disparity <= 1

