import pandas as pd
from sklearn.model_selection import train_test_split
from aif360.datasets import RegressionDataset
from aif360.sklearn.datasets.lawschool_dataset import LSAC_URL

class LawSchoolGPADataset(RegressionDataset):
    """Law School GPA dataset."""

    def __init__(self, dep_var_name='zfygpa',
                 protected_attribute_names=['race', 'gender'],
                 privileged_classes=[['white'], ['male']],
                 instance_weights_name=None,
                 categorical_features=[],
                 na_values=[], custom_preprocessing=None,
                 metadata=None):
        """See :obj:`RegressionDataset` for a description of the arguments."""
        df = pd.read_sas(LSAC_URL, encoding="utf-8")
        df.race = df.race1.where(df.race1.isin(['black', 'white']))
        df.gender = df.gender.fillna("female")
        df = df[["race", "gender", "lsat", "ugpa", "zfygpa"]].dropna()
        df = pd.concat(train_test_split(df, test_size=0.33, random_state=123))

        super(LawSchoolGPADataset, self).__init__(df=df,
            dep_var_name=dep_var_name,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
