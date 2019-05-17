# Original work Copyright 2017 Flavio Calmon
# Modified work Copyright 2018 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
from warnings import warn

import numpy as np
import pandas as pd

from aif360.algorithms import Transformer
from aif360.datasets import BinaryLabelDataset


class OptimPreproc(Transformer):
    """Optimized preprocessing is a preprocessing technique that learns a
    probabilistic transformation that edits the features and labels in the data
    with group fairness, individual distortion, and data fidelity constraints
    and objectives [3]_.

    References:
        .. [3] F. P. Calmon, D. Wei, B. Vinzamuri, K. Natesan Ramamurthy, and
           K. R. Varshney. "Optimized Pre-Processing for Discrimination
           Prevention." Conference on Neural Information Processing Systems,
           2017.

    Based on code available at: https://github.com/fair-preprocessing/nips2017
    """

    def __init__(self, optimizer, optim_options, unprivileged_groups=None,
                 privileged_groups=None, verbose=False, seed=None):
        """
        Args:
            optimizer (class): Optimizer class.
            optim_options (dict): Options for optimization to estimate the
                transformation.
            unprivileged_groups (dict): Representation for unprivileged group.
            privileged_groups (dict): Representation for privileged group.
            verbose (bool, optional): Verbosity flag for optimization.
            seed (int, optional): Seed to make `fit` and `predict` repeatable.

        Note:
            This algorithm does not use the privileged and unprivileged groups
            that are specified during initialization yet. Instead, it
            automatically attempts to reduce statistical parity difference
            between all possible combinations of groups in the dataset.
        """

        super(OptimPreproc, self).__init__(optimizer=optimizer,
            optim_options=optim_options,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups, verbose=verbose, seed=seed)

        self.seed = seed
        self.optimizer = optimizer
        self.optim_options = optim_options
        self.verbose = verbose

        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups
        if unprivileged_groups or privileged_groups:
            warn("Privileged and unprivileged groups specified will not be "
                 "used. The protected attributes are directly specified in the "
                 "data preprocessing function. The current implementation "
                 "automatically adjusts for discrimination across all groups. "
                 "This can be changed by changing the optimization code.")

    def fit(self, dataset, sep='='):
        """Compute optimal pre-processing transformation based on distortion
        constraint.

        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.
            sep (str, optional): Separator for converting one-hot labels to
                categorical.
        Returns:
            OptimPreproc: Returns self.
        """
        if len(np.unique(dataset.instance_weights)) > 1:
            warn("Optimized pre-processing will ignore instance_weights in "
                 "the dataset during fit.")
        # Convert the dataset to a dataframe and preprocess
        df, _ = dataset.convert_to_dataframe(de_dummy_code=True, sep=sep,
                                             set_category=True)

        # Subset the protected attribute names and attribute values from
        # input parameters
        self.protected_attribute_names = dataset.protected_attribute_names
        self.privileged_protected_attributes = dataset.privileged_protected_attributes
        self.unprivileged_protected_attributes = dataset.unprivileged_protected_attributes

        # Feature names
        self.Y_feature_names = dataset.label_names
        self.X_feature_names = [n for n in df.columns.tolist()
                                if n not in self.Y_feature_names
                                and n not in self.protected_attribute_names]
        self.feature_names = (self.X_feature_names + self.Y_feature_names
                            + self.protected_attribute_names)

        # initialize a new OptTools object
        self.OpT = self.optimizer(df=df, features=self.feature_names)

        # Set features
        self.OpT.set_features(D=self.protected_attribute_names,
                              X=self.X_feature_names,
                              Y=self.Y_feature_names)

        # Set Distortion
        self.OpT.set_distortion(self.optim_options['distortion_fun'],
                                clist=self.optim_options['clist'])

        # solve optimization for previous parameters
        self.OpT.optimize(epsilon=self.optim_options['epsilon'],
                          dlist=self.optim_options['dlist'],
                          verbose=self.verbose)

        # Compute marginals
        self.OpT.compute_marginals()

        return self

    def transform(self, dataset, sep='=', transform_Y=True):
        """Transform the dataset to a new dataset based on the estimated
        transformation.

        Args:
            dataset (BinaryLabelDataset): Dataset containing labels that needs
                to be transformed.
            transform_Y (bool): Flag that mandates transformation of Y (labels).
        """

        if len(np.unique(dataset.instance_weights)) > 1:
            warn("Optimized pre-processing will ignore instance_weights in "
                 "the dataset during predict. The transformed dataset will "
                 "have all instance weights set to 1.")

        # Convert the dataset to a dataframe and preprocess
        df, _ = dataset.convert_to_dataframe(de_dummy_code=True, sep=sep,
                                             set_category=True)

        # Feature names
        Y_feature_names = dataset.label_names
        D_feature_names = self.protected_attribute_names
        X_feature_names = [n for n in df.columns.tolist()
                           if n not in self.Y_feature_names
                           and n not in self.protected_attribute_names]

        if (X_feature_names != self.X_feature_names or
            D_feature_names != self.protected_attribute_names):

           raise ValueError("The feature names of inputs and protected "
                            "attributes must match with the training dataset.")

        if transform_Y and (Y_feature_names != self.Y_feature_names):
            raise ValueError("The label names must match with that in the training dataset")

        if transform_Y:
            # randomized mapping when Y is requested to be transformed
            dfP_withY = self.OpT.dfP.applymap(lambda x: 0 if x < 1e-8 else x)
            dfP_withY = dfP_withY.divide(dfP_withY.sum(axis=1), axis=0)

            df_transformed = _apply_randomized_mapping(df, dfP_withY,
                features=D_feature_names+X_feature_names+Y_feature_names,
                random_seed=self.seed)
        else:
            # randomized mapping when Y is not requested to be transformed
            d1 = self.OpT.dfFull.reset_index().groupby(
                D_feature_names+X_feature_names).sum()
            d2 = d1.transpose().reset_index().groupby(X_feature_names).sum()
            dfP_noY = d2.transpose()
            dfP_noY = dfP_noY.drop(Y_feature_names, 1)
            dfP_noY = dfP_noY.applymap(lambda x: x if x > 1e-8 else 0)
            dfP_noY = dfP_noY/dfP_noY.sum()

            dfP_noY = dfP_noY.divide(dfP_noY.sum(axis=1), axis=0)

            df_transformed = _apply_randomized_mapping(
                                df, dfP_noY,
                                features=D_feature_names+X_feature_names,
                                random_seed=self.seed)

        # Map the protected attributes to numeric values
        for idx, p in enumerate(self.protected_attribute_names):
            pmap = dataset.metadata["protected_attribute_maps"][idx]
            pmap_rev = dict(zip(pmap.values(), pmap.keys()))
            df_transformed[p] = df_transformed[p].replace(pmap_rev)

        # Map the labels to numeric values
        for idx, p in enumerate(Y_feature_names):
            pmap = dataset.metadata["label_maps"][idx]
            pmap_rev = dict(zip(pmap.values(), pmap.keys()))
            df_transformed[p] = df_transformed[p].replace(pmap_rev)

        # Dummy code and convert to a dataset
        df_dum = pd.concat([pd.get_dummies(df_transformed.loc[:, X_feature_names],
                            prefix_sep="="),
                            df_transformed.loc[:, Y_feature_names+D_feature_names]],
                            axis=1)

        # Create a dataset out of df_dum
        dataset_transformed = BinaryLabelDataset(
            df=df_dum,
            label_names=Y_feature_names,
            protected_attribute_names=self.protected_attribute_names,
            privileged_protected_attributes=self.privileged_protected_attributes,
            unprivileged_protected_attributes=self.unprivileged_protected_attributes,
            favorable_label=dataset.favorable_label,
            unfavorable_label=dataset.unfavorable_label,
            metadata=dataset.metadata)

        return dataset_transformed

    def fit_transform(self, dataset, sep='=', transform_Y=True):
        """Perfom :meth:`fit` and :meth:`transform` sequentially."""

        return self.fit(dataset, sep=sep).transform(dataset, sep=sep,
                                                    transform_Y=transform_Y)

##############################
#### Supporting functions ####
##############################
def _apply_randomized_mapping(df, dfMap,
                              features=[], random_seed=None):
    """Apply Randomized mapping to create a new dataframe

    Args:
        df (DataFrame): Input dataframe
        dfMap (DataFrame): Mapping parameters
        features (list): Feature names for which the mapping needs to be applied
        random_seed (int): Random seed

    Returns:
        Perturbed version of df according to the randomizedmapping
    """

    if random_seed is not None:
        np.random.seed(seed=random_seed)

    df2 = df[features].copy()
    rem_cols = [l for l in df.columns
                if l not in features]
    if rem_cols != []:
        df3 = df[rem_cols].copy()

    idx_list = [tuple(i) for i in df2.itertuples(index=False)]

    draw_probs = dfMap.loc[idx_list]
    draws_possible = draw_probs.columns.tolist()

    # Make random draws - as part of randomizing transformation
    def draw_ind(x): return np.random.choice(range(len(draws_possible)), p=x)

    draw_inds = [draw_ind(x) for x in draw_probs.values]

    df2.loc[:, dfMap.columns.names] = [draws_possible[x] for x in draw_inds]

    if rem_cols != []:
        return pd.concat([df2, df3], axis=1)
    else:
        return df2
