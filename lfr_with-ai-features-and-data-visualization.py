import os
import sys

import numpy as np
import pytest
import matplotlib.pyplot as plt  # For visualization
import seaborn as sns  # For advanced data visualization

from aif360.algorithms.preprocessing import LFR
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult

sys.path.append("../../../")
sys.path.append(os.path.dirname(__file__))

# AI-Driven Enhancements: Visualization function for analyzing distribution before and after LFR transformation
def visualize_data_distribution(dataset, title="Data Distribution"):
    """Visualize the distribution of features in the dataset."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=dataset.features, kde=True)
    plt.title(title)
    plt.xlabel("Feature Values")
    plt.ylabel("Frequency")
    plt.show()

@pytest.fixture(scope="module")
def lfr_algorithm_instance():
    """Fixture to create an instance of the LFR algorithm."""
    privileged_groups = [{'sex': 1.0}]
    unprivileged_groups = [{'sex': 0.0}]

    lfr_instance = LFR(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    return lfr_instance

@pytest.fixture(scope="module")
def adult_data():
    """Fixture to load and split the adult dataset."""
    return load_preproc_data_adult().split([0.7], shuffle=True)[0]

@pytest.fixture(scope="module")
def lfr_fit_model():
    """Fixture to fit the LFR model on the adult dataset."""
    privileged_groups = [{'sex': 1.0}]
    unprivileged_groups = [{'sex': 0.0}]

    lfr_instance = LFR(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    ad = load_preproc_data_adult().split([0.7], shuffle=True)[0]

    # AI Enhancement: Visualize the original data distribution before fitting
    visualize_data_distribution(ad, title="Original Data Distribution")

    transformed_data = lfr_instance.fit(ad)

    # AI Enhancement: Visualize the transformed data distribution after fitting
    visualize_data_distribution(transformed_data, title="Transformed Data Distribution (After LFR Fit)")

    return transformed_data

def test_fit_is_numpy(lfr_fit_model):
    """Test that the fit function returns a numpy array."""
    expected = True
    result = isinstance(lfr_fit_model.learned_model, np.ndarray)
    assert result == expected

def test_fit_not_null(lfr_fit_model):
    """Test that the fit function's result is not null."""
    expected = False
    result = lfr_fit_model.learned_model is None
    assert result == expected

def test_fit_not_all_zeros(lfr_fit_model):
    """Test that the fit function's result is not all zeros."""
    expected = False
    all_zeros = not np.any(lfr_fit_model.learned_model)
    assert all_zeros == expected

def test_fit_not_nan(lfr_fit_model):
    """Test that the fit function's result does not contain NaNs."""
    expected = False
    result = np.isnan(lfr_fit_model.learned_model).any()
    assert result == expected

# --------------------------------------------------------------------#
# Transform function testing methods
# --------------------------------------------------------------------#

def test_transform_protected_dataset(lfr_fit_model, adult_data):
    """Test that the protected attributes remain unchanged after transformation."""
    transformed_data = lfr_fit_model.transform(adult_data, threshold=0.3)
    result = np.array_equal(transformed_data.protected_attributes, adult_data.protected_attributes)
    assert result == True

    # AI Enhancement: Visualize the protected attribute distribution after transformation
    visualize_data_distribution(transformed_data, title="Protected Attributes After LFR Transformation")

def test_transform_not_nan(lfr_fit_model, adult_data):
    """Test that the transformed data does not contain NaNs and does not have columns or rows summing up to zero."""
    transformed_data = lfr_fit_model.transform(adult_data, threshold=0.3)
    row_sums = np.sum(transformed_data.features, axis=1).tolist()
    col_sums = np.sum(transformed_data.features, axis=0).tolist()
    all_row_zeros = not np.any(row_sums)
    all_col_zeros = not np.any(col_sums)
    assert (all_row_zeros and all_col_zeros) == False

def test_transform_not_nan2(lfr_fit_model, adult_data):
    """Test that the transformed data does not contain NaNs with a threshold value of 0.3."""
    transformed_data = lfr_fit_model.transform(adult_data, threshold=0.3)
    result = np.isnan(transformed_data.features).any()
    assert result == False

    # AI Enhancement: Visualize the transformed data distribution for NaN checking
    visualize_data_distribution(transformed_data, title="Transformed Data Distribution (No NaNs)")
