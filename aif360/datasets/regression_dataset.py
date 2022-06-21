from logging import warning

import numpy as np
import pandas as pd

from aif360.datasets import StructuredDataset

from sklearn.preprocessing import MinMaxScaler


class RegressionDataset(StructuredDataset):
    """Base class for regression datasets."""

    def __init__(self, df, dep_var_name, protected_attribute_names,
                 privileged_classes, instance_weights_name='',
                 categorical_features=[], na_values=[],
                 custom_preprocessing=None, metadata=None):
        """
        Subclasses of RegressionDataset should perform the following before
        calling `super().__init__`:

            1. Load the dataframe from a raw file.

        Then, this class will go through a standard preprocessing routine which:

            2. (optional) Performs some dataset-specific preprocessing (e.g.
               renaming columns/values, handling missing data).

            3. Drops rows with NA values.

            4. Creates a one-hot encoding of the categorical variables.

            5. Maps protected attributes to binary privileged/unprivileged
               values (1/0).

            6. Normalizes df values

        Args:
            df (pandas.DataFrame): DataFrame on which to perform standard
                processing.
            dep_var_name: Name of the dependent variable column in `df`.
            protected_attribute_names (list): List of names corresponding to
                protected attribute columns in `df`.
            privileged_classes (list(list or function)): Each element is
                a list of values which are considered privileged or a boolean
                function which return `True` if privileged for the corresponding
                column in `protected_attribute_names`. All others are
                unprivileged. Values are mapped to 1 (privileged) and 0
                (unprivileged) if they are not already numerical.
            instance_weights_name (optional): Name of the instance weights
                column in `df`.
            categorical_features (optional, list): List of column names in the
                DataFrame which are to be expanded into one-hot vectors.
            na_values (optional): Additional strings to recognize as NA. See
                :func:`pandas.read_csv` for details.
            custom_preprocessing (function): A function object which
                acts on and returns a DataFrame (f: DataFrame -> DataFrame). If
                `None`, no extra preprocessing is applied.
            metadata (optional): Additional metadata to append.
        """
        # 2. Perform dataset-specific preprocessing
        if custom_preprocessing:
            df = custom_preprocessing(df)

        # 3. Remove any rows that have missing data.
        dropped = df.dropna()
        count = df.shape[0] - dropped.shape[0]
        if count > 0:
            warning("Missing Data: {} rows removed from {}.".format(count,
                    type(self).__name__))
        df = dropped

        # 4. Create a one-hot encoding of the categorical variables.
        df = pd.get_dummies(df, columns=categorical_features, prefix_sep='=')

        # 5. Map protected attributes to privileged/unprivileged
        privileged_protected_attributes = []
        unprivileged_protected_attributes = []
        for attr, vals in zip(protected_attribute_names, privileged_classes):
            privileged_values = [1.]
            unprivileged_values = [0.]
            if callable(vals):
                df[attr] = df[attr].apply(vals)
            elif np.issubdtype(df[attr].dtype, np.number):
                # this attribute is numeric; no remapping needed
                privileged_values = vals
                unprivileged_values = list(set(df[attr]).difference(vals))
            else:
                # find all instances which match any of the attribute values
                priv = np.logical_or.reduce(np.equal.outer(vals, df[attr].to_numpy()))
                df.loc[priv, attr] = privileged_values[0]
                df.loc[~priv, attr] = unprivileged_values[0]

            privileged_protected_attributes.append(
                np.array(privileged_values, dtype=np.float64))
            unprivileged_protected_attributes.append(
                np.array(unprivileged_values, dtype=np.float64))

        # 6. Normalize df values
        df = pd.DataFrame(MinMaxScaler().fit_transform(df.values),
                          columns=list(df), index=df.index)

        super(RegressionDataset, self).__init__(df=df,
            label_names=[dep_var_name],
            protected_attribute_names=protected_attribute_names,
            privileged_protected_attributes=privileged_protected_attributes,
            unprivileged_protected_attributes=unprivileged_protected_attributes,
            instance_weights_name=instance_weights_name,
            scores_names=[],
            metadata=metadata)
