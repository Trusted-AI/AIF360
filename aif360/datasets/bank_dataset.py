from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import pandas as pd

from aif360.datasets import StandardDataset


class BankDataset(StandardDataset):
    """Bank marketing Dataset.

    See :file:`aif360/data/raw/bank/README.md`.
    """

    def __init__(self, label_name='y', favorable_classes=['yes'],
                 protected_attribute_names=['age'],
                 privileged_classes=[lambda x: x >= 25],
                 instance_weights_name=None,
                 categorical_features=['job', 'marital', 'education', 'default',
                     'housing', 'loan', 'contact', 'month', 'day_of_week',
                     'poutcome'],
                 features_to_keep=[], features_to_drop=[],
                 na_values=["unknown"], custom_preprocessing=None,
                 metadata=None):
        """See :obj:`StandardDataset` for a description of the arguments.

        By default, this code converts the 'age' attribute to a binary value
        where privileged is `age >= 25` and unprivileged is `age < 25` as in
        :obj:`GermanDataset`.
        """

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../data/raw/bank/bank-additional-full.csv')
        df = pd.read_csv(filepath, sep=';', na_values=na_values)

        super(BankDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
