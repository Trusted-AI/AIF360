import os
import sys

import numpy as np
import pytest

from aif360.algorithms.preprocessing import LFR
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult


sys.path.append("../../../")
sys.path.append(os.path.dirname(__file__))


@pytest.fixture(scope="module")
def lfrAlgoInstance():
    """This fixture creates two functions with the scope module lfrAlgoInstance creates an instance of the LFR that can
     used by both fit and transform functions.
    ad creates a adult data set that will be used by the fit and the transform functions.
    """
    privileged_groups = [{'sex': 1.0}]
    unprivileged_groups = [{'sex': 0.0}]

    lfrAlgoInstance = LFR(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    return lfrAlgoInstance


@pytest.fixture(scope="module")
def ad():
    return load_preproc_data_adult().split([0.7], shuffle=True)[0]


@pytest.fixture(scope="module")
def lfrfitmodel():
    privileged_groups = [{'sex': 1.0}]
    unprivileged_groups = [{'sex': 0.0}]

    lfrAlgoInstance = LFR(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    ad = load_preproc_data_adult().split([0.7], shuffle=True)[0]
    TR = lfrAlgoInstance.fit(ad)

    return TR


def test_fit_isnumpy(lfrfitmodel):
    """The Fit function returns a numpy and it should asserted whether it really returned a numpy precision 64 bits.
    """
    expected = True
    if type(lfrfitmodel.learned_model) is np.ndarray:
        res = True
    else:
        res = False
    assert res == expected


def test_fit_notnull(lfrfitmodel):
    """Should not be null.
    """
    expected = False
    if lfrfitmodel.learned_model is None:
        res = True
    else:
        res = False
    print("numpy:" + str(res))
    assert res == expected


def test_fit_notallzeros(lfrfitmodel):
    """Should not be all zeros.
    """
    expected = False
    all_zeros = not np.any(lfrfitmodel)
    print("allzeros:" + str(all_zeros))
    assert all_zeros == expected


def test_fit_notNaN(lfrfitmodel):
    """Should not have nan's in it.
    """
    expected = False
    res = np.isnan(lfrfitmodel.learned_model).any()
    print("nan:" + str(res))
    assert res == expected


# --------------------------------------------------------------------#
# Transform function testing methods
# --------------------------------------------------------------------#

def test_transform_protecteddataset(lfrfitmodel, ad):
    """After transformation - it should not change protected attributes - it should be same as input.
    """
    lftransformeddataset = lfrfitmodel.transform(ad, threshold=0.3)
    # print( ad.protected_attributes)
    print(type(lftransformeddataset.protected_attributes))
    # print("transformeddataset:" + lfttransformeddataset.protected_attributes)
    expected = True
    res = np.array_equal(lftransformeddataset.protected_attributes, ad.protected_attributes)
    assert res == expected


def test_transform_notNaN(lfrfitmodel, ad):
    """The transformed data should not have any columns or rows summing upto zero.
    """
    lftransformeddataset = lfrfitmodel.transform(ad, threshold=0.3)
    lstrowsum = np.sum(lftransformeddataset.features, axis=1).tolist()
    expected = False
    allrow_zeros = not np.any(lstrowsum)
    lstcolsum = np.sum(lftransformeddataset.features, axis=0).tolist()
    allcol_zeros = not np.any(lstcolsum)
    assert (allrow_zeros and allcol_zeros) == expected


def test_transform_notNaN2(lfrfitmodel, ad):
    """The transformed data should not contain nan's. Using the threshold value of 0.3.
    """
    lftransformeddataset = lfrfitmodel.transform(ad, threshold=0.3)
    expected = False
    res = np.isnan(lftransformeddataset.features).any()
    assert res == expected
