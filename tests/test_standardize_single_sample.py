"""Offline public dataset preparation must preserve a one-row sample axis."""
import os
from pathlib import Path
import sys
import unittest


SOURCE_ROOT = Path(os.environ.get(
    'AIF360_SOURCE_ROOT', str(Path(__file__).resolve().parents[1])
)).resolve()
sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.linear_model import LinearRegression

from aif360.sklearn.datasets import standardize_dataset


class StandardizeSingleSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        actual = Path(sys.modules[standardize_dataset.__module__].__file__).resolve()
        expected = SOURCE_ROOT / 'aif360' / 'sklearn' / 'datasets' / 'utils.py'
        if actual != expected:
            raise AssertionError('Wrong dataset source: {} != {}'.format(actual, expected))

    def frame(self, target=1):
        return pd.DataFrame({'feature': [3.5], 'group': [1], 'target': [target]})

    def assert_single(self, result, target=1):
        self.assertIsInstance(result.X, pd.DataFrame)
        self.assertIsInstance(result.y, pd.Series)
        self.assertEqual(result.X.shape[0], 1)
        self.assertEqual(result.y.shape, (1,))
        expected = pd.Series([target], index=pd.Index([1], name='group'), name='target')
        assert_series_equal(result.y, expected)
        self.assertTrue(result.X.index.equals(result.y.index))

    def test_single_sample_preserves_series_and_feature_values(self):
        result = standardize_dataset(self.frame(), prot_attr='group', target='target')
        self.assert_single(result)
        expected = pd.DataFrame({'feature': [3.5], 'group': [1]},
                                index=pd.Index([1], name='group'))
        assert_frame_equal(result.X, expected)
        self.assertEqual(result._fields, ('X', 'y'))

    def test_single_sample_preserves_target_values_and_types(self):
        for value in (True, 2.5, 'positive'):
            with self.subTest(value=value):
                result = standardize_dataset(self.frame(value), prot_attr='group', target='target')
                self.assert_single(result, value)

    def test_single_sample_with_external_target_series(self):
        frame = self.frame().drop(columns='target')
        target = pd.Series([7], index=[99], name='outcome')
        result = standardize_dataset(frame, prot_attr='group', target=target)
        expected = pd.Series([7], index=pd.Index([1], name='group'), name='outcome')
        assert_series_equal(result.y, expected)
        self.assertTrue(result.X.index.equals(result.y.index))
        self.assertEqual(result.X.columns.tolist(), ['feature', 'group'])

    def test_single_sample_with_weight_column(self):
        frame = self.frame()
        frame['weight'] = [2.5]
        result = standardize_dataset(frame, prot_attr='group', target='target', sample_weight='weight')
        self.assert_single(result)
        assert_series_equal(result.sample_weight,
                            pd.Series([2.5], index=result.X.index, name='weight'))
        self.assertNotIn('weight', result.X.columns)
        self.assertEqual(result._fields, ('X', 'y', 'sample_weight'))

    def test_single_sample_with_external_weight_array(self):
        result = standardize_dataset(self.frame(), prot_attr='group', target='target',
                                     sample_weight=np.array([0.75]))
        self.assert_single(result)
        assert_series_equal(result.sample_weight, pd.Series([0.75], index=result.X.index))

    def test_numeric_only_and_feature_selection_preserve_one_sample(self):
        frame = self.frame()
        frame['text'] = ['description']
        result = standardize_dataset(frame, prot_attr='group', target='target',
                                     numeric_only=True, usecols=['feature', 'text'])
        self.assert_single(result)
        self.assertEqual(result.X.columns.tolist(), ['feature'])
        self.assertEqual(result.X.iloc[0, 0], 3.5)

    def test_multiple_protected_attributes_preserve_one_sample(self):
        frame = self.frame()
        frame['region'] = [0]
        result = standardize_dataset(frame, prot_attr=['group', 'region'], target='target')
        expected_index = pd.MultiIndex.from_tuples([(1, 0)], names=['group', 'region'])
        assert_series_equal(result.y, pd.Series([1], index=expected_index, name='target'))
        self.assertTrue(result.X.index.equals(expected_index))

    def test_single_sample_multiple_targets_preserve_the_row_axis(self):
        frame = self.frame()
        frame['other_target'] = [2]
        result = standardize_dataset(frame, prot_attr='group',
                                     target=['target', 'other_target'], dropna=False)
        expected = pd.DataFrame({'target': [1], 'other_target': [2]},
                                index=pd.Index([1], name='group'))
        assert_frame_equal(result.y, expected)
        self.assertTrue(result.X.index.equals(result.y.index))

    def test_dropna_removes_a_single_missing_target_without_losing_container(self):
        result = standardize_dataset(self.frame(np.nan), prot_attr='group', target='target')
        self.assertIsInstance(result.y, pd.Series)
        self.assertEqual(result.X.shape, (0, 2))
        self.assertEqual(result.y.shape, (0,))
        self.assertTrue(result.X.index.equals(result.y.index))

    def test_dropna_removes_a_single_missing_feature_and_weight_consistently(self):
        frame = self.frame()
        frame['feature'] = [np.nan]
        frame['weight'] = [2.5]
        result = standardize_dataset(frame, prot_attr='group', target='target', sample_weight='weight')
        self.assertEqual(result.X.shape[0], 0)
        self.assertIsInstance(result.y, pd.Series)
        self.assertEqual(len(result.y), 0)
        self.assertEqual(len(result.sample_weight), 0)
        self.assertTrue(result.X.index.equals(result.sample_weight.index))

    def test_documented_estimator_handoff_accepts_single_sample(self):
        frame = self.frame(2.5)
        frame['weight'] = [2.0]
        train = standardize_dataset(frame, prot_attr='group', target='target', sample_weight='weight')
        model = LinearRegression().fit(**train._asdict())
        np.testing.assert_allclose(model.predict(train.X), [2.5])

    def test_multiple_samples_control(self):
        frame = pd.DataFrame({'feature': [3.5, 4.5], 'group': [1, 0], 'target': [0, 1]})
        result = standardize_dataset(frame, prot_attr='group', target='target')
        expected = pd.Series([0, 1], index=pd.Index([1, 0], name='group'), name='target')
        assert_series_equal(result.y, expected)
        self.assertEqual(result.X.shape, (2, 2))

    def test_zero_sample_control(self):
        result = standardize_dataset(self.frame().iloc[:0], prot_attr='group', target='target')
        self.assertIsInstance(result.y, pd.Series)
        self.assertEqual(result.X.shape, (0, 2))
        self.assertEqual(result.y.shape, (0,))
        self.assertTrue(result.X.index.equals(result.y.index))


if __name__ == '__main__':
    unittest.main()
