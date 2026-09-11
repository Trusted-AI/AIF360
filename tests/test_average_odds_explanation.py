import json

import pandas as pd

from aif360.datasets import BinaryLabelDataset
from aif360.explainers import MetricJSONExplainer, MetricTextExplainer
from aif360.metrics import ClassificationMetric


def test_average_odds_cancellation():
    # Group 0 has TPR=0 and FPR=1; group 1 has TPR=1 and FPR=0.
    # Their signed differences cancel, despite unequal odds.
    data = BinaryLabelDataset(
        df=pd.DataFrame({'group': [0, 0, 1, 1], 'label': [0, 1, 0, 1]}),
        label_names=['label'], protected_attribute_names=['group'])
    predicted = data.copy(deepcopy=True)
    predicted.labels = 1 - data.labels
    predicted.labels[2:] = data.labels[2:]
    metric = ClassificationMetric(data, predicted,
        unprivileged_groups=[{'group': 0}], privileged_groups=[{'group': 1}])

    assert metric.average_odds_difference() == 0
    assert metric.average_abs_odds_difference() == 1
    assert metric.equalized_odds_difference() == 1

    message = MetricTextExplainer(metric).average_odds_difference()
    assert '0 does not imply equality of odds' in message
    explanation = json.loads(MetricJSONExplainer(metric).average_odds_difference())
    assert explanation['message'] == message
    assert 'cancel' in explanation['ideal']
