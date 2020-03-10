from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from logging import warning

import numpy as np
import pandas as pd

from aif360.datasets import Dataset


class StructuredDataset(Dataset):
    """Base class for all structured datasets.

    A StructuredDataset requires data to be stored in :obj:`numpy.ndarray`
    objects with :obj:`~numpy.dtype` as :obj:`~numpy.float64`.

    Attributes:
        features (numpy.ndarray): Dataset features for each instance.
        labels (numpy.ndarray): Generic label corresponding to each instance
            (could be ground-truth, predicted, cluster assignments, etc.).
        scores (numpy.ndarray): Probability score associated with each label.
            Same shape as `labels`. Only valid for binary labels (this includes
            one-hot categorical labels as well).
        protected_attributes (numpy.ndarray): A subset of `features` for which
            fairness is desired.
        feature_names (list(str)): Names describing each dataset feature.
        label_names (list(str)): Names describing each label.
        protected_attribute_names (list(str)): A subset of `feature_names`
            corresponding to `protected_attributes`.
        privileged_protected_attributes (list(numpy.ndarray)): A subset of
            protected attribute values which are considered privileged from a
            fairness perspective.
        unprivileged_protected_attributes (list(numpy.ndarray)): The remaining
            possible protected attribute values which are not included in
            `privileged_protected_attributes`.
        instance_names (list(str)): Indentifiers for each instance. Sequential
            integers by default.
        instance_weights (numpy.ndarray):  Weighting for each instance. All
            equal (ones) by default. Pursuant to standard practice in social
            science data, 1 means one person or entity. These weights are hence
            person or entity multipliers (see:
            https://www.ibm.com/support/knowledgecenter/en/SS3RA7_15.0.0/com.ibm.spss.modeler.help/netezza_decisiontrees_weights.htm)
            These weights *may not* be normalized to sum to 1 across the entire
            dataset, rather the nominal (default) weight of each entity/record
            in the data is 1. This is similar in spirit to the person weight in
            census microdata samples.
            https://www.census.gov/programs-surveys/acs/technical-documentation/pums/about.html
        ignore_fields (set(str)): Attribute names to ignore when doing equality
            comparisons. Always at least contains `'metadata'`.
        metadata (dict): Details about the creation of this dataset. For
            example::

                {
                    'transformer': 'Dataset.__init__',
                    'params': kwargs,
                    'previous': None
                }
    """

    def __init__(self, df, label_names, protected_attribute_names,
                 instance_weights_name=None, scores_names=[],
                 unprivileged_protected_attributes=[],
                 privileged_protected_attributes=[], metadata=None):
        """
        Args:
            df (pandas.DataFrame): Input DataFrame with features, labels, and
                protected attributes. Values should be preprocessed
                to remove NAs and make all data numerical. Index values are
                taken as instance names.
            label_names (iterable): Names of the label columns in `df`.
            protected_attribute_names (iterable): List of names corresponding to
                protected attribute columns in `df`.
            instance_weights_name (optional): Column name in `df` corresponding
                to instance weights. If not provided, `instance_weights` will be
                all set to 1.
            unprivileged_protected_attributes (optional): If not provided, all
                but the highest numerical value of each protected attribute will
                be considered not privileged.
            privileged_protected_attributes (optional): If not provided, the
                highest numerical value of each protected attribute will be
                considered privileged.
            metadata (optional): Additional metadata to append.

        Raises:
            TypeError: Certain fields must be np.ndarrays as specified in the
                class description.
            ValueError: ndarray shapes must match.
        """
        if df is None:
            raise TypeError("Must provide a pandas DataFrame representing "
                            "the data (features, labels, protected attributes)")
        if df.isna().any().any():
            raise ValueError("Input DataFrames cannot contain NA values.")
        try:
            df = df.astype(np.float64)
        except ValueError as e:
            print("ValueError: {}".format(e))
            raise ValueError("DataFrame values must be numerical.")

        # Convert all column names to strings
        df.columns = df.columns.astype(str).tolist()
        label_names = list(map(str, label_names))
        protected_attribute_names = list(map(str, protected_attribute_names))

        self.feature_names = [n for n in df.columns if n not in label_names
                              and (not scores_names or n not in scores_names)
                              and n != instance_weights_name]
        self.label_names = label_names
        self.features = df[self.feature_names].values.copy()
        self.labels = df[self.label_names].values.copy()
        self.instance_names = df.index.astype(str).tolist()

        if scores_names:
            self.scores = df[scores_names].values.copy()
        else:
            self.scores = self.labels.copy()

        df_prot = df.loc[:, protected_attribute_names]
        self.protected_attribute_names = df_prot.columns.astype(str).tolist()
        self.protected_attributes = df_prot.values.copy()

        # Infer the privileged and unprivileged values in not provided
        if unprivileged_protected_attributes and privileged_protected_attributes:
            self.unprivileged_protected_attributes = unprivileged_protected_attributes
            self.privileged_protected_attributes = privileged_protected_attributes
        else:
            self.unprivileged_protected_attributes = [
                np.sort(np.unique(df_prot[attr].values))[:-1]
                for attr in self.protected_attribute_names]
            self.privileged_protected_attributes = [
                np.sort(np.unique(df_prot[attr].values))[-1:]
                for attr in self.protected_attribute_names]

        if instance_weights_name:
            self.instance_weights = df[instance_weights_name].values.copy()
        else:
            self.instance_weights = np.ones_like(self.instance_names,
                dtype=np.float64)

        # always ignore metadata and ignore_fields
        self.ignore_fields = {'metadata', 'ignore_fields'}

        # sets metadata
        super(StructuredDataset, self).__init__(df=df, label_names=label_names,
            protected_attribute_names=protected_attribute_names,
            instance_weights_name=instance_weights_name,
            unprivileged_protected_attributes=unprivileged_protected_attributes,
            privileged_protected_attributes=privileged_protected_attributes,
            metadata=metadata)


    def subset(self, indexes):
        """ Subset of dataset based on position
        Args:
            indexes: iterable which contains row indexes

        Returns:
            `StructuredDataset`: subset of dataset based on indexes
        """
        # convert each element of indexes to string
        indexes_str = [self.instance_names[i] for i in indexes]
        subset = self.copy()
        subset.instance_names = indexes_str
        subset.features = self.features[indexes]
        subset.labels = self.labels[indexes]
        subset.instance_weights = self.instance_weights[indexes]
        subset.protected_attributes = self.protected_attributes[indexes]
        subset.scores = self.scores[indexes]
        return subset


    def __eq__(self, other):
        """Equality comparison for StructuredDatasets.

        Note: Compares all fields other than those specified in `ignore_fields`.
        """
        if not isinstance(other, StructuredDataset):
            return False

        def _eq(x, y):
            if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                return np.all(x == y)
            elif isinstance(x, list) and isinstance(y, list):
                return len(x) == len(y) and all(_eq(xi, yi) for xi, yi in zip(x, y))
            return x == y

        return all(_eq(self.__dict__[k], other.__dict__[k])
                   for k in self.__dict__.keys() if k not in self.ignore_fields)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        # return repr(self.metadata)
        return str(self)

    def __str__(self):
        df, _ = self.convert_to_dataframe()
        df.insert(0, 'instance_weights', self.instance_weights)
        highest_level = ['instance weights'] + \
                        ['features']*len(self.feature_names) + \
                        ['labels']*len(self.label_names)
        middle_level = [''] + \
                       ['protected attribute'
                           if f in self.protected_attribute_names else ''
                           for f in self.feature_names] + \
                       ['']*len(self.label_names)
        lowest_level = [''] + self.feature_names + ['']*len(self.label_names)
        df.columns = pd.MultiIndex.from_arrays(
            [highest_level, middle_level, lowest_level])
        df.index.name = 'instance names'
        return str(df)

    # TODO: *_names checks
    def validate_dataset(self):
        """Error checking and type validation.

        Raises:
            TypeError: Certain fields must be np.ndarrays as specified in the
                class description.
            ValueError: ndarray shapes must match.
        """
        super(StructuredDataset, self).validate_dataset()

        # =========================== TYPE CHECKING ============================
        for f in [self.features, self.protected_attributes, self.labels,
                  self.scores, self.instance_weights]:
            if not isinstance(f, np.ndarray):
                raise TypeError("'{}' must be an np.ndarray.".format(f.__name__))

        # convert ndarrays to float64
        self.features = self.features.astype(np.float64)
        self.protected_attributes = self.protected_attributes.astype(np.float64)
        self.labels = self.labels.astype(np.float64)
        self.instance_weights = self.instance_weights.astype(np.float64)

        # =========================== SHAPE CHECKING ===========================
        if len(self.labels.shape) == 1:
            self.labels = self.labels.reshape((-1, 1))
        try:
            self.scores.reshape(self.labels.shape)
        except ValueError as e:
            print("ValueError: {}".format(e))
            raise ValueError("'scores' should have the same shape as 'labels'.")
        if not self.labels.shape[0] == self.features.shape[0]:
            raise ValueError("Number of labels must match number of instances:"
                "\n\tlabels.shape = {}\n\tfeatures.shape = {}".format(
                    self.labels.shape, self.features.shape))
        if not self.instance_weights.shape[0] == self.features.shape[0]:
            raise ValueError("Number of weights must match number of instances:"
                "\n\tinstance_weights.shape = {}\n\tfeatures.shape = {}".format(
                    self.instance_weights.shape, self.features.shape))

        # =========================== VALUE CHECKING ===========================
        if np.any(np.logical_or(self.scores < 0., self.scores > 1.)):
            warning("'scores' has no well-defined meaning out of range [0, 1].")

        for i in range(len(self.privileged_protected_attributes)):
            priv = set(self.privileged_protected_attributes[i])
            unpriv = set(self.unprivileged_protected_attributes[i])
            # check for duplicates
            if priv & unpriv:
                raise ValueError("'privileged_protected_attributes' and "
                    "'unprivileged_protected_attributes' should not share any "
                    "common elements:\n\tBoth contain {} for feature {}".format(
                        list(priv & unpriv), self.protected_attribute_names[i]))
            # check for unclassified values
            if not set(self.protected_attributes[:, i]) <= (priv | unpriv):
                raise ValueError("All observed values for protected attributes "
                    "should be designated as either privileged or unprivileged:"
                    "\n\t{} not designated for feature {}".format(
                        list(set(self.protected_attributes[:, i])
                           - (priv | unpriv)),
                        self.protected_attribute_names[i]))
            # warn for unobserved values
            if not (priv | unpriv) <= set(self.protected_attributes[:, i]):
                warning("{} listed but not observed for feature {}".format(
                    list((priv | unpriv) - set(self.protected_attributes[:, i])),
                    self.protected_attribute_names[i]))

    @contextmanager
    def temporarily_ignore(self, *fields):
        """Temporarily add the fields provided to `ignore_fields`.

        To be used in a `with` statement. Upon completing the `with` block,
        `ignore_fields` is restored to its original value.

        Args:
            *fields: Additional fields to ignore for equality comparison within
                the scope of this context manager, e.g.
                `temporarily_ignore('features', 'labels')`. The temporary
                `ignore_fields` attribute is the union of the old attribute and
                the set of these fields.

        Examples:
            >>> sd = StructuredDataset(...)
            >>> modified = sd.copy()
            >>> modified.labels = sd.labels + 1
            >>> assert sd != modified
            >>> with sd.temporarily_ignore('labels'):
            >>>     assert sd == modified
            >>> assert 'labels' not in sd.ignore_fields
        """
        old_ignore = deepcopy(self.ignore_fields)
        self.ignore_fields |= set(fields)
        try:
            yield
        finally:
            self.ignore_fields = old_ignore

    def align_datasets(self, other):
        """Align the other dataset features, labels and protected_attributes to
        this dataset.

        Args:
            other (StructuredDataset): Other dataset that needs to be aligned

        Returns:
            StructuredDataset: New aligned dataset
        """

        if (set(self.feature_names) != set(other.feature_names) or
            set(self.label_names) != set(other.label_names) or
            set(self.protected_attribute_names)
                != set(other.protected_attribute_names)):
            raise ValueError(
                "feature_names, label_names, and protected_attribute_names "
                "should match between this and other dataset.")

        # New dataset
        new = other.copy()

        # re-order the columns of the new dataset
        feat_inds = [new.feature_names.index(f) for f in self.feature_names]
        label_inds = [new.label_names.index(f) for f in self.label_names]
        prot_inds = [new.protected_attribute_names.index(f)
                     for f in self.protected_attribute_names]

        new.features = new.features[:, feat_inds]
        new.labels = new.labels[:, label_inds]
        new.scores = new.scores[:, label_inds]
        new.protected_attributes = new.protected_attributes[:, prot_inds]

        new.privileged_protected_attributes = [
            new.privileged_protected_attributes[i] for i in prot_inds]
        new.unprivileged_protected_attributes = [
            new.unprivileged_protected_attributes[i] for i in prot_inds]
        new.feature_names = deepcopy(self.feature_names)
        new.label_names = deepcopy(self.label_names)
        new.protected_attribute_names = deepcopy(self.protected_attribute_names)

        return new

    # TODO: Should we store the protected attributes as a separate dataframe
    def convert_to_dataframe(self, de_dummy_code=False, sep='=',
                             set_category=True):
        """Convert the StructuredDataset to a :obj:`pandas.DataFrame`.

        Args:
            de_dummy_code (bool): Performs de_dummy_coding, converting dummy-
                coded columns to categories. If `de_dummy_code` is `True` and
                this dataset contains mappings for label and/or protected
                attribute values to strings in the `metadata`, this method will
                convert those as well.
            sep (char): Separator between the prefix in the dummy indicators and
                the dummy-coded categorical levels.
            set_category (bool): Set the de-dummy coded features to categorical
                type.

        Returns:
            (pandas.DataFrame, dict):

                * `pandas.DataFrame`: Equivalent dataframe for a dataset. All
                  columns will have only numeric values. The
                  `protected_attributes` field in the dataset will override the
                  values in the `features` field.

                * `dict`: Attributes. Will contain additional information pulled
                  from the dataset such as `feature_names`, `label_names`,
                  `protected_attribute_names`, `instance_names`,
                  `instance_weights`, `privileged_protected_attributes`,
                  `unprivileged_protected_attributes`. The metadata will not be
                  returned.

        """
        df = pd.DataFrame(np.hstack((self.features, self.labels)),
            columns=self.feature_names+self.label_names,
            index=self.instance_names)
        df.loc[:, self.protected_attribute_names] = self.protected_attributes

        # De-dummy code if necessary
        if de_dummy_code:
            df = self._de_dummy_code_df(df, sep=sep, set_category=set_category)
            if 'label_maps' in self.metadata:
                for i, label in enumerate(self.label_names):
                    df[label] = df[label].replace(self.metadata['label_maps'][i])
            if 'protected_attribute_maps' in self.metadata:
                for i, prot_attr in enumerate(self.protected_attribute_names):
                    df[prot_attr] = df[prot_attr].replace(
                        self.metadata['protected_attribute_maps'][i])

        # Attributes
        attributes = {
            "feature_names": self.feature_names,
            "label_names": self.label_names,
            "protected_attribute_names": self.protected_attribute_names,
            "instance_names": self.instance_names,
            "instance_weights": self.instance_weights,
            "privileged_protected_attributes": self.privileged_protected_attributes,
            "unprivileged_protected_attributes": self.unprivileged_protected_attributes
        }

        return df, attributes

    def export_dataset(self, export_metadata=False):
        """
        Export the dataset and supporting attributes
        TODO: The preferred file format is HDF
        """

        if export_metadata:
            raise NotImplementedError("The option to export metadata has not been implemented yet")

        return None

    def import_dataset(self, import_metadata=False):
        """ Import the dataset and supporting attributes
            TODO: The preferred file format is HDF
        """

        if import_metadata:
            raise NotImplementedError("The option to import metadata has not been implemented yet")
        return None

    def split(self, num_or_size_splits, shuffle=False, seed=None):
        """Split this dataset into multiple partitions.

        Args:
            num_or_size_splits (array or int): If `num_or_size_splits` is an
                int, *k*, the value is the number of equal-sized folds to make
                (if *k* does not evenly divide the dataset these folds are
                approximately equal-sized). If `num_or_size_splits` is an array
                of type int, the values are taken as the indices at which to
                split the dataset. If the values are floats (< 1.), they are
                considered to be fractional proportions of the dataset at which
                to split.
            shuffle (bool, optional): Randomly shuffle the dataset before
                splitting.
            seed (int or array_like): Takes the same argument as
                :func:`numpy.random.seed()`.

        Returns:
            list: Splits. Contains *k* or `len(num_or_size_splits) + 1`
            datasets depending on `num_or_size_splits`.
        """

        # Set seed
        if seed is not None:
            np.random.seed(seed)

        n = self.features.shape[0]
        if isinstance(num_or_size_splits, list):
            num_folds = len(num_or_size_splits) + 1
            if num_folds > 1 and all(x <= 1. for x in num_or_size_splits):
                num_or_size_splits = [int(x * n) for x in num_or_size_splits]
        else:
            num_folds = num_or_size_splits

        order = list(np.random.permutation(n) if shuffle else range(n))
        folds = [self.copy() for _ in range(num_folds)]

        features = np.array_split(self.features[order], num_or_size_splits)
        labels = np.array_split(self.labels[order], num_or_size_splits)
        scores = np.array_split(self.scores[order], num_or_size_splits)
        protected_attributes = np.array_split(self.protected_attributes[order],
            num_or_size_splits)
        instance_weights = np.array_split(self.instance_weights[order],
            num_or_size_splits)
        instance_names = np.array_split(np.array(self.instance_names)[order],
            num_or_size_splits)
        for fold, feats, labs, scors, prot_attrs, inst_wgts, inst_name in zip(
                folds, features, labels, scores, protected_attributes, instance_weights,
                instance_names):

            fold.features = feats
            fold.labels = labs
            fold.scores = scors
            fold.protected_attributes = prot_attrs
            fold.instance_weights = inst_wgts
            fold.instance_names = list(map(str, inst_name))
            fold.metadata = fold.metadata.copy()
            fold.metadata.update({
                'transformer': '{}.split'.format(type(self).__name__),
                'params': {'num_or_size_splits': num_or_size_splits,
                           'shuffle': shuffle},
                'previous': [self]
            })

        return folds

    @staticmethod
    def _de_dummy_code_df(df, sep="=", set_category=False):
        """De-dummy code a dummy-coded dataframe obtained with pd.get_dummies().

        After reversing dummy coding the corresponding fields will be converted
        to categorical.

        Args:
            df (pandas.DataFrame): Input dummy coded dataframe
            sep (char): Separator between base name and dummy code
            set_category (bool): Set the de-dummy coded features
                    to categorical type

        Examples:
            >>> columns = ["Age", "Gender=Male", "Gender=Female"]
            >>> df = pd.DataFrame([[10, 1, 0], [20, 0, 1]], columns=columns)
            >>> _de_dummy_code_df(df, sep="=")
               Age  Gender
            0   10    Male
            1   20  Female
        """

        feature_names_dum_d, feature_names_nodum = \
            StructuredDataset._parse_feature_names(df.columns)
        df_new = pd.DataFrame(index=df.index,
            columns=feature_names_nodum + list(feature_names_dum_d.keys()))

        for fname in feature_names_nodum:
            df_new[fname] = df[fname].values.copy()

        for fname, vl in feature_names_dum_d.items():
            for v in vl:
                df_new.loc[df[fname+sep+str(v)] == 1, fname] = str(v)

        if set_category:
            for fname in feature_names_dum_d.keys():
                df_new[fname] = df_new[fname].astype('category')

        return df_new

    @staticmethod
    def _parse_feature_names(feature_names, sep="="):
        """Parse feature names to ordinary and dummy coded candidates.

        Args:
            feature_names (list): Names of features
            sep (char): Separator to designate the dummy coded category in the
                feature name

        Returns:
            (dict, list):

                * feature_names_dum_d (dict): Keys are the base feature names
                  and values are the categories.

                * feature_names_nodum (list): Non-dummy coded feature names.

        Examples:
            >>> feature_names = ["Age", "Gender=Male", "Gender=Female"]
            >>> StructuredDataset._parse_feature_names(feature_names, sep="=")
            (defaultdict(<type 'list'>, {'Gender': ['Male', 'Female']}), ['Age'])
        """
        feature_names_dum_d = defaultdict(list)
        feature_names_nodum = list()
        for fname in feature_names:
            if sep in fname:
                fname_dum, v = fname.split(sep, 1)
                feature_names_dum_d[fname_dum].append(v)
            else:
                feature_names_nodum.append(fname)

        return feature_names_dum_d, feature_names_nodum
