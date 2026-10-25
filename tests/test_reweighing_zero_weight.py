"""Real Reweighing regressions; no downloaded data or estimator substitutes."""
import os
from pathlib import Path
import sys
import unittest

SOURCE_ROOT = Path(os.environ.get(
    "AIF360_SOURCE_ROOT", Path(__file__).resolve().parents[1]
)).resolve()
sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import LogisticRegression

import aif360.sklearn.preprocessing.reweighing as reweighing_module
from aif360.sklearn.preprocessing import Reweighing, ReweighingMeta

if Path(reweighing_module.__file__).resolve() != (
    SOURCE_ROOT / "aif360/sklearn/preprocessing/reweighing.py"
).resolve():
    raise RuntimeError("Regression imported the wrong AIF360 source tree")


def frame(groups, values):
    return pd.DataFrame(
        {"feature": values}, index=pd.Index(groups, name="group")
    )


class TestZeroWeightReweighing(unittest.TestCase):
    def test_zero_mass_cell_matches_removing_zero_weight_rows(self):
        X = frame([0, 0, 1, 1], [-2., 1., -1., 2.])
        y = np.array([0, 1, 0, 1])
        for weights in ([1, 1, 1, 0], np.array([1., 1., 1., 0.]),
                        pd.Series([1., 1., 1., 0.], index=X.index)):
            with self.subTest(weight_type=type(weights).__name__):
                before_X = X.copy(deep=True)
                before_weights = np.asarray(weights).copy()
                keep = before_weights > 0
                rew = Reweighing("group")
                with np.errstate(divide="ignore", invalid="ignore"):
                    returned_X, actual = rew.fit_transform(X, y, weights)
                _, removed = Reweighing("group").fit_transform(
                    X.iloc[keep], y[keep], before_weights[keep]
                )
                self.assertIs(returned_X, X)
                self.assertTrue(np.isfinite(actual).all())
                assert_allclose(actual[keep], removed)
                assert_allclose(actual, [4 / 3, 2 / 3, 2 / 3, 0])
                self.assertTrue(np.isnan(rew.reweigh_factors_[1, 1]))
                pd.testing.assert_frame_equal(X, before_X)
                assert_array_equal(np.asarray(weights), before_weights)

    def test_group_with_only_zero_weight_rows(self):
        X = frame([0, 0, 1, 1], [-2., 1., -1., 2.])
        y = np.array([0, 1, 0, 1])
        rew = Reweighing("group")
        with np.errstate(divide="ignore", invalid="ignore"):
            _, actual = rew.fit_transform(X, y, [1, 1, 0, 0])
        assert_allclose(actual, [1, 1, 0, 0])
        self.assertTrue(np.isnan(rew.reweigh_factors_[1]).all())

    def test_class_with_only_zero_weight_rows(self):
        X = frame([0, 0, 1, 1], [-2., 1., -1., 2.])
        y = np.array([0, 1, 0, 1])
        rew = Reweighing("group")
        with np.errstate(divide="ignore", invalid="ignore"):
            _, actual = rew.fit_transform(X, y, [1, 0, 1, 0])
        assert_allclose(actual, [1, 0, 1, 0])
        self.assertTrue(np.isnan(rew.reweigh_factors_[:, 1]).all())

    def test_fractional_weights_and_permuted_rows(self):
        X = frame([0, 0, 1, 1, 1], [-2., 1., -1., 2., 3.])
        y = np.array([0, 1, 0, 1, 1])
        weights = np.array([.5, 1.5, 2., 0., 0.])
        keep = weights > 0
        _, expected_positive = Reweighing("group").fit_transform(
            X.iloc[keep], y[keep], weights[keep]
        )
        expected = np.zeros(len(y))
        expected[keep] = expected_positive
        order = np.array([4, 2, 0, 3, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            _, actual = Reweighing("group").fit_transform(
                X.iloc[order], y[order], weights[order]
            )
        assert_allclose(actual, expected[order])

    def test_meta_passes_finite_weights_to_real_logistic_regression(self):
        X = frame([0, 0, 1, 1], [-2., 1., -1., 2.])
        y = np.array([0, 1, 0, 1])
        weights = np.array([1., 1., 1., 0.])
        keep = weights > 0
        _, removed = Reweighing("group").fit_transform(
            X.iloc[keep], y[keep], weights[keep]
        )
        reference = LogisticRegression(
            solver="liblinear", random_state=0, tol=1e-10
        ).fit(X.iloc[keep], y[keep], sample_weight=removed)
        for reweigher in (None, Reweighing("group")):
            with self.subTest(default_reweigher=reweigher is None):
                meta = ReweighingMeta(
                    LogisticRegression(
                        solver="liblinear", random_state=0, tol=1e-10
                    ), reweigher
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    result = meta.fit(X, y, sample_weight=weights)
                self.assertIs(result, meta)
                self.assertIsInstance(meta.estimator_, LogisticRegression)
                assert_allclose(meta.estimator_.coef_, reference.coef_,
                                rtol=1e-7, atol=1e-9)
                assert_allclose(meta.estimator_.intercept_, reference.intercept_,
                                rtol=1e-7, atol=1e-9)
                assert_allclose(meta.predict_proba(X), reference.predict_proba(X),
                                rtol=1e-7, atol=1e-9)

    def test_positive_weight_formula_unchanged(self):
        X = frame([0, 0, 1, 1], [-2., 1., -1., 2.])
        _, actual = Reweighing("group").fit_transform(
            X, [0, 1, 0, 1], [1, 2, 3, 4]
        )
        assert_allclose(actual, [6 / 5, 9 / 5, 14 / 5, 21 / 5])

    def test_absent_cell_factor_remains_nan(self):
        X = frame([0, 0, 1], [-2., 1., -1.])
        rew = Reweighing("group")
        _, actual = rew.fit_transform(X, [0, 1, 0], [1, 1, 1])
        assert_allclose(actual, [4 / 3, 2 / 3, 2 / 3])
        self.assertTrue(np.isnan(rew.reweigh_factors_[1, 1]))

    def test_zero_weight_in_positive_mass_cell_unchanged(self):
        X = frame([0, 0, 1, 1, 0], [-2., 1., -1., 2., -3.])
        _, actual = Reweighing("group").fit_transform(
            X, [0, 1, 0, 1, 0], [2, 2, 2, 2, 0]
        )
        assert_allclose(actual, [2, 2, 2, 2, 0])


if __name__ == "__main__":
    unittest.main()
