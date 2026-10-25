"""Offline real LogisticRegression -> PostProcessingMeta -> calibrated odds.

No Adult fixture, provider, fabricated estimator or fairness-algorithm stub.
Main selects a complete source checkout via AIF360_SOURCE_ROOT.
"""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SOURCE_ROOT = Path(os.environ.get(
    'AIF360_SOURCE_ROOT', str(Path(__file__).resolve().parents[1]))).resolve()
sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd
from pandas.testing import assert_index_equal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from aif360.sklearn.postprocessing import CalibratedEqualizedOdds, PostProcessingMeta


class PostProcessingMetaArrayFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        actual = Path(sys.modules[PostProcessingMeta.__module__].__file__).resolve()
        expected = SOURCE_ROOT / 'aif360' / 'sklearn' / 'postprocessing' / '__init__.py'
        if actual != expected:
            raise AssertionError('Wrong source module: {} != {}'.format(actual, expected))

    def data(self):
        rows = np.array([[-2., 0.], [-1., 1.], [.2, -1.], [1.3, 2.],
                         [-.4, -2.], [.8, -1.], [1.2, .4], [2., 1.]])
        X = np.tile(rows, (12, 1))
        groups = np.tile([0, 0, 0, 0, 1, 1, 1, 1], 12)
        index = pd.MultiIndex.from_arrays(
            [groups, ['case-{}'.format(11 + 7 * i) for i in range(len(X))]],
            names=['group', 'row'])
        frame = pd.DataFrame(X, index=index, columns=['first', 'second'])
        y = pd.Series(np.tile([0, 0, 0, 1, 0, 1, 1, 1], 12), index=index)
        weights = np.tile([1., 2., 1., 3., 4., 1., 2., 1.], 12)
        return frame, y, weights

    def fit_meta(self, X, y, weights, prefit, shuffle):
        estimator = LogisticRegression(solver='lbfgs', max_iter=500, random_state=13)
        if prefit:
            estimator.fit(X, y, sample_weight=weights)
        postprocessor = CalibratedEqualizedOdds('group', random_state=23)
        meta = PostProcessingMeta(
            estimator, postprocessor, prefit=prefit, val_size=.5,
            **({'shuffle': shuffle, 'random_state': 17} if not prefit else {}))
        observed = []
        real_fit = CalibratedEqualizedOdds.fit

        def observe_real_fit(instance, scores, targets, *args, **kwargs):
            # Transparent observation; execute the complete real postprocessor.
            observed.append((scores.index.copy(), targets.index.copy()))
            return real_fit(instance, scores, targets, *args, **kwargs)

        with patch.object(CalibratedEqualizedOdds, 'fit', observe_real_fit):
            self.assertIs(meta.fit(X, y, sample_weight=weights), meta)
        self.assertEqual(len(observed), 1)
        self.assertIsInstance(meta.estimator_, LogisticRegression)
        self.assertIsInstance(meta.postprocessor_, CalibratedEqualizedOdds)
        if prefit:
            self.assertIs(meta.estimator_, estimator)
        if prefit:
            expected_index = y.index
        else:
            _, expected_y = train_test_split(
                y, test_size=.5, shuffle=shuffle, random_state=17)
            expected_index = expected_y.index
        assert_index_equal(observed[0][0], expected_index)
        assert_index_equal(observed[0][1], expected_index)
        return meta

    def assert_matches_dataframe(self, prefit, weighted, shuffle=False):
        frame, y, weights = self.data()
        selected_weights = weights if weighted else None
        array = frame.to_numpy(copy=True)
        untouched = array.copy()
        from_array = self.fit_meta(array, y, selected_weights, prefit, shuffle)
        reference = self.fit_meta(frame, y, selected_weights, prefit, shuffle)
        np.testing.assert_array_equal(array, untouched)
        np.testing.assert_array_equal(from_array.classes_, reference.classes_)
        np.testing.assert_array_equal(from_array.postprocessor_.groups_, reference.postprocessor_.groups_)
        np.testing.assert_allclose(from_array.postprocessor_.base_rates_, reference.postprocessor_.base_rates_)
        np.testing.assert_allclose(from_array.postprocessor_.mix_rates_, reference.postprocessor_.mix_rates_)
        for fitted, features in ((from_array, array), (reference, frame)):
            self.assertTrue(np.isfinite(fitted.postprocessor_.mix_rates_).all())
            scores = pd.DataFrame(fitted.estimator_.predict_proba(features), index=frame.index)
            probabilities = fitted.postprocessor_.predict_proba(scores)
            if fitted is from_array:
                actual = probabilities
            else:
                np.testing.assert_allclose(actual, probabilities)

    def test_prefit_array_uses_label_index(self):
        self.assert_matches_dataframe(prefit=True, weighted=False)

    def test_prefit_weighted_array_uses_label_index(self):
        self.assert_matches_dataframe(prefit=True, weighted=True)

    def test_split_array_uses_matching_split_label_index(self):
        self.assert_matches_dataframe(prefit=False, weighted=False)

    def test_shuffled_weighted_split_array_keeps_group_alignment(self):
        self.assert_matches_dataframe(prefit=False, weighted=True, shuffle=True)

    def test_prefit_dataframe_control(self):
        frame, y, _ = self.data()
        self.fit_meta(frame, y, None, prefit=True, shuffle=False)

    def test_split_weighted_dataframe_control(self):
        frame, y, weights = self.data()
        self.fit_meta(frame, y, weights, prefit=False, shuffle=True)


if __name__ == '__main__':
    unittest.main()
