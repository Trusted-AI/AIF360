"""Offline public-API regressions for the alpha=0 mean log deviation.

Set AIF360_SOURCE_ROOT to select an uninstalled source tree. Without it this
file uses its repository root. All AIF360 classes and calculations are real;
no dataset download, model training, algorithm mock or benefit injection.
"""
import math
import os
from pathlib import Path
import sys
import unittest


SOURCE_ROOT = Path(os.environ.get(
    "AIF360_SOURCE_ROOT", str(Path(__file__).resolve().parents[1])
)).resolve()
sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.sklearn.metrics import (
    between_group_generalized_entropy_error,
    generalized_entropy_error,
    generalized_entropy_index,
)


class GeneralizedEntropyAlphaZeroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Fail instead of accidentally testing a different installed checkout.
        for implementation in (BinaryLabelDataset, ClassificationMetric,
                               generalized_entropy_index):
            module = sys.modules[implementation.__module__]
            actual = Path(module.__file__).resolve()
            expected = SOURCE_ROOT.joinpath(*module.__name__.split(".")).with_suffix(".py")
            if actual != expected:
                raise AssertionError("Wrong source module: {} != {}".format(actual, expected))

    def assert_entropy(self, actual, expected):
        self.assertTrue(np.isfinite(actual), "Expected a finite entropy")
        self.assertAlmostEqual(float(actual), expected, places=12)

    def classification_metric(self, y_true, y_pred, groups):
        frame = pd.DataFrame({"group": groups, "label": y_true})
        truth = BinaryLabelDataset(
            df=frame, label_names=["label"],
            protected_attribute_names=["group"],
            favorable_label=1, unfavorable_label=0,
        )
        prediction = truth.copy(deepcopy=True)
        prediction.labels = np.asarray(y_pred, dtype=np.float64).reshape(-1, 1)
        prediction.scores = prediction.labels.copy()
        return ClassificationMetric(
            truth, prediction,
            unprivileged_groups=[{"group": 0}],
            privileged_groups=[{"group": 1}],
        )

    def grouped_fixture(self):
        groups = [0, 0, 1, 1]
        y_true = pd.Series([0, 0, 1, 0], index=pd.Index(groups, name="group"))
        y_pred = np.array([1, 1, 1, 0])
        metric = self.classification_metric(y_true.to_numpy(), y_pred, groups)
        return y_true, y_pred, metric

    def test_positive_benefits_match_closed_form(self):
        # Ratios [1/2, 1, 3/2]; product 3/4, hence -mean(log)=log(4/3)/3.
        actual = generalized_entropy_index(np.array([1., 2., 3.]), alpha=0)
        self.assert_entropy(actual, math.log(4. / 3.) / 3.)

    def test_positive_scaling_preserves_closed_form(self):
        expected = math.log(4. / 3.) / 3.
        base = np.array([1., 2., 3.])
        reference = generalized_entropy_index(base, alpha=0)
        for scale in (0.25, 2., 10.):
            with self.subTest(scale=scale):
                actual = generalized_entropy_index(scale * base, alpha=0)
                self.assert_entropy(actual, expected)
                self.assert_entropy(actual, float(reference))

    def test_unit_mean_nonuniform_control(self):
        # This already works on the old implementation, where the extra /mu is /1.
        actual = generalized_entropy_index(np.array([0.5, 1., 1.5]), alpha=0)
        self.assert_entropy(actual, math.log(4. / 3.) / 3.)

    def test_equal_positive_benefits_control(self):
        for alpha in (0, 1, 2):
            for value in (1., 4.25):
                with self.subTest(alpha=alpha, value=value):
                    self.assert_entropy(generalized_entropy_index(
                        np.full(4, value), alpha=alpha), 0.)

    def test_sklearn_label_entry_matches_closed_form(self):
        # Real labels generate [2,1,1], with no log(0) or probability inputs.
        actual = generalized_entropy_error(
            np.array([0, 0, 1]), np.array([1, 0, 1]), alpha=0)
        self.assert_entropy(actual, math.log(32. / 27.) / 3.)

    def test_sklearn_public_group_entries_match_closed_form(self):
        y_true, y_pred, _ = self.grouped_fixture()
        # Group means are [2,2,1,1]; normalized ratios [4/3,4/3,2/3,2/3].
        for priv_group in (None, 1):
            with self.subTest(priv_group=priv_group):
                actual = between_group_generalized_entropy_error(
                    y_true, y_pred, prot_attr="group",
                    priv_group=priv_group, alpha=0)
                self.assert_entropy(actual, math.log(9. / 8.) / 2.)

    def test_legacy_individual_matches_closed_form(self):
        metric = self.classification_metric([0, 0, 1], [1, 0, 1], [0, 0, 1])
        self.assert_entropy(metric.generalized_entropy_index(alpha=0),
                            math.log(32. / 27.) / 3.)

    def test_legacy_public_between_group_matches_closed_form(self):
        _, _, metric = self.grouped_fixture()
        self.assert_entropy(metric.between_group_generalized_entropy_index(alpha=0),
                            math.log(9. / 8.) / 2.)

    def test_legacy_public_between_all_groups_matches_closed_form(self):
        _, _, metric = self.grouped_fixture()
        self.assert_entropy(metric.between_all_groups_generalized_entropy_index(alpha=0),
                            math.log(9. / 8.) / 2.)

    def check_control_alpha(self, alpha, expected):
        y_true, y_pred, metric = self.grouped_fixture()
        results = {
            "direct": generalized_entropy_index(np.array([2., 2., 1., 1.]), alpha=alpha),
            "sklearn_labels": generalized_entropy_error(y_true.to_numpy(), y_pred, alpha=alpha),
            "sklearn_groups": between_group_generalized_entropy_error(
                y_true, y_pred, prot_attr="group", alpha=alpha),
            "sklearn_priv_group": between_group_generalized_entropy_error(
                y_true, y_pred, prot_attr="group", priv_group=1, alpha=alpha),
            "legacy_individual": metric.generalized_entropy_index(alpha=alpha),
            "legacy_between": metric.between_group_generalized_entropy_index(alpha=alpha),
            "legacy_between_all": metric.between_all_groups_generalized_entropy_index(alpha=alpha),
        }
        for entry, actual in results.items():
            with self.subTest(entry=entry):
                self.assert_entropy(actual, expected)

    def test_alpha_one_controls(self):
        expected = (2. / 3.) * math.log(4. / 3.) + (1. / 3.) * math.log(2. / 3.)
        self.check_control_alpha(1, expected)

    def test_alpha_two_controls(self):
        self.check_control_alpha(2, 1. / 18.)

    def test_perfect_prediction_controls(self):
        groups = [0, 0, 1, 1]
        y_true = pd.Series([0, 1, 0, 1], index=pd.Index(groups, name="group"))
        y_pred = y_true.to_numpy().copy()
        metric = self.classification_metric(y_pred, y_pred, groups)
        for alpha in (0, 1, 2):
            with self.subTest(alpha=alpha):
                self.assert_entropy(generalized_entropy_error(y_pred, y_pred, alpha=alpha), 0.)
                self.assert_entropy(between_group_generalized_entropy_error(
                    y_true, y_pred, prot_attr="group", alpha=alpha), 0.)
                self.assert_entropy(metric.generalized_entropy_index(alpha=alpha), 0.)
                self.assert_entropy(metric.between_group_generalized_entropy_index(alpha=alpha), 0.)
                self.assert_entropy(metric.between_all_groups_generalized_entropy_index(alpha=alpha), 0.)


if __name__ == "__main__":
    unittest.main()
