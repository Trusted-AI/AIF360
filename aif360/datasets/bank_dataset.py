import os

import pandas as pd

from aif360.datasets import StandardDataset


class BankDataset(StandardDataset):
    """Bank marketing Dataset.

    See :file:`aif360/data/raw/bank/README.md`.
    """

    def __init__(self, label_name='y', favorable_classes=['yes'],
                 protected_attribute_names=['age'],
                 privileged_classes=[lambda x: x >= 25 and x < 60],
                 instance_weights_name=None,
                 categorical_features=['job', 'marital', 'education', 'default',
                     'housing', 'loan', 'contact', 'month', 'day_of_week',
                     'poutcome'],
                 features_to_keep=[], features_to_drop=[],
                 na_values=["unknown"], custom_preprocessing=None,
                 metadata=None):
        """See :obj:`StandardDataset` for a description of the arguments.

        By default, this code converts the 'age' attribute to a binary value
        where privileged is `25 <= age < 60` and unprivileged is `age < 25` or `age >= 60`
        as suggested in Le Quy, Tai, et al. [1].

        References:
            .. [1] Le Quy, Tai, et al. "A survey on datasets for fairness‐aware machine 
            learning." Wiley Interdisciplinary Reviews: Data Mining and Knowledge 
            Discovery 12.3 (2022): e1452.


        """

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '..', 'data', 'raw', 'bank', 'bank-additional-full.csv')

        try:
            df = pd.read_csv(filepath, sep=';', na_values=na_values)
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following file:")
            print("\n\thttps://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip")
            print("\nunzip it and place the files, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', '..', 'data', 'raw', 'bank'))))
            import sys
            sys.exit(1)

        super(BankDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
