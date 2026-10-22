import numpy as np
import pandas as pd

from aif360.datasets import MulticlassLabelDataset
from aif360.datasets.multiclass_label_dataset import (
    MulticlassLabelDataset as MulticlassLabelDatasetFromModule)


def test_importable_from_datasets_package():
    assert MulticlassLabelDataset is MulticlassLabelDatasetFromModule


def test_init_from_package_import():
    df = pd.DataFrame({'feat': [0, 1, 2, 1], 'label': [0, 1, 2, 2]})
    mcld = MulticlassLabelDataset(favorable_label=[0, 1], unfavorable_label=[2],
                                  df=df, label_names=['label'],
                                  protected_attribute_names=['feat'])

    assert np.all(mcld.labels == df[['label']].values)
    assert mcld.favorable_label == [0, 1]
    assert mcld.unfavorable_label == [2]
