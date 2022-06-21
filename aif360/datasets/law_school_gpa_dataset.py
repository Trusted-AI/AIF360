import os
import pandas as pd
from aif360.datasets import RegressionDataset
import tempeh.configurations as tc


class LawSchoolGPADataset(RegressionDataset):
    """Law School GPA dataset.

    See https://github.com/microsoft/tempeh for details.
    """

    def __init__(self, dep_var_name='zfygpa',
                 protected_attribute_names=['race'],
                 privileged_classes=[['white']],
                 instance_weights_name=None,
                 categorical_features=[],
                 na_values=[], custom_preprocessing=None,
                 metadata=None):
        """See :obj:`RegressionDataset` for a description of the arguments."""
        dataset = tc.datasets["lawschool_gpa"]()
        X_train,X_test = dataset.get_X(format=pd.DataFrame)
        y_train, y_test = dataset.get_y(format=pd.Series)
        A_train, A_test = dataset.get_sensitive_features(name='race',
                                                         format=pd.Series)
        all_train = pd.concat([X_train, y_train, A_train], axis=1)
        all_test = pd.concat([X_test, y_test, A_test], axis=1)

        df = pd.concat([all_train, all_test], axis=0)

        super(LawSchoolGPADataset, self).__init__(df=df,
            dep_var_name=dep_var_name,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
