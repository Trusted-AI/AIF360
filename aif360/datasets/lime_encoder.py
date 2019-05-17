import sklearn.preprocessing
import numpy as np

from aif360.algorithms import Transformer


class LimeEncoder(Transformer):
    """Tranformer for converting aif360 dataset to LIME dataset and vice versa.

    (LIME - Local Interpretable Model-Agnostic Explanations) [2]_

    See for details/usage:
    https://github.com/marcotcr/lime

    References:
        .. [2] M.T. Ribeiro, S. Singh, and C. Guestrin, '"Why should I trust
           you?" Explaining the predictions of any classifier.'
           https://arxiv.org/pdf/1602.04938v1.pdf
    """

    def __init__(self):
        super(LimeEncoder, self).__init__()

    def fit(self, dataset):
        """Take an aif360 dataset and save all relevant metadata as well as
        mappings needed to transform/inverse_transform the data between aif360
        and lime.

        Args:
            dataset (BinaryLabelDataset): aif360 dataset

        Returns:
            LimeEncoder: Returns self.
        """
        self.s_feature_names_with_one_hot_encoding = dataset.feature_names
        df, df_dict = dataset.convert_to_dataframe(de_dummy_code=True)

        dfc = df.drop(dataset.label_names[0], axis=1)  # remove label (class) column

        self.s_feature_names = list(dfc.columns)       # create list of feature names
        self.s_data = dfc.values                       # create array of feature values

        # since categorical features are 1-hot-encoded and their names changed,
        # the set diff gives us the list of categorical features as non-
        # categorical feature names are not changed
        self.s_categorical_features = list(set(self.s_feature_names)
                                         - set(self.s_feature_names_with_one_hot_encoding))

        self.s_protected_attribute_names = dataset.protected_attribute_names

        # add protected attribute names to the list of categorical features
        self.s_categorical_features = self.s_categorical_features \
                                    + self.s_protected_attribute_names

        self.s_labels = df[dataset.label_names[0]]  # create labels

        # following 3 lines are not really needed
        # using to create s_class_names..can do so manually as well ...array([ 0.,  1.])
        s_le = sklearn.preprocessing.LabelEncoder()
        s_le.fit(self.s_labels)
        # self.s_labels = s_le.transform(self.s_labels)
        self.s_class_names = s_le.classes_

        # convert s_categorical_features to a list of array indexes in
        # s_feature_names corresponding to categorical features
        # (NOTE - does not included protected attributes)
        self.s_categorical_features = [self.s_feature_names.index(x)
                                       for x in self.s_categorical_features]

        # map all the categorical features to numerical values and store the
        # mappings in s_categorical_names
        self.s_categorical_names = {}
        for feature in self.s_categorical_features:
            self.le = sklearn.preprocessing.LabelEncoder()
            self.le.fit(self.s_data[:, feature])
            #self.s_data[:, feature] = le.transform(self.s_data[:, feature])
            self.s_categorical_names[feature] = self.le.classes_

        return self

    def transform(self, aif360data):
        """Take aif360 data array and return data array that is lime encoded
        (numeric array in which categorical features are NOT one-hot-encoded).

        Args:
            aif360data (np.ndarray): Dataset features

        Returns:
            np.ndarray: LIME dataset features
        """
        tgtNumRows = aif360data.shape[0]
        tgtNumcolumns = len(self.s_feature_names)
        limedata = np.zeros(shape=(tgtNumRows, tgtNumcolumns))

        # non_categorical_features = list(set(self.s_feature_names) & set(self.s_feature_names_with_one_hot_encoding))
        for rw in range(limedata.shape[0]):
            for ind, feature in enumerate(self.s_feature_names):
                if ind in self.s_categorical_features:
                    # tranform the value since categorical feature except if it
                    # is also a protected attribute
                    if feature in self.s_protected_attribute_names:
                        # just copy the value as is
                        limedata[rw, ind] = aif360data[rw, self.s_feature_names_with_one_hot_encoding.index(feature)]
                    else:
                        possible_feature_values = self.s_categorical_names[ind]
                        for indc in range(len(possible_feature_values)):
                            cval = possible_feature_values[indc]
                            colName = feature + "=" + cval
                            if (aif360data[rw][self.s_feature_names_with_one_hot_encoding.index(colName)] == 1.0):
                                limedata[rw][ind] = indc
                else:
                    # just copy the value as is
                    limedata[rw, ind] = aif360data[rw, self.s_feature_names_with_one_hot_encoding.index(feature)]

        return limedata

    def inverse_transform(self, limedata):
        """Take data array that is lime encoded (that is, lime-compatible data
        created by this class from a given aif360 dataset) and return data array
        consistent with the original aif360 dataset.

        Args:
            limedata (np.ndarray): Dataset features

        Returns:
            np.ndarray: aif360 dataset features
        """
        tgtNumRows = limedata.shape[0]
        tgtNumcolumns = len(self.s_feature_names_with_one_hot_encoding)
        aif360data = np.zeros(shape=(tgtNumRows, tgtNumcolumns))

        for rw in range(aif360data.shape[0]):
            for ind, feature in enumerate(self.s_feature_names):
                # s_categorical_features has list of indexes into
                # s_feature_names for categorical features
                if ind in self.s_categorical_features:
                    if feature in self.s_protected_attribute_names:
                        # just copy the value as is
                        aif360data[rw, self.s_feature_names_with_one_hot_encoding.index(feature)] = limedata[rw, ind]
                    else:
                        # s_categorical_names[ind] has mapping of categorical to
                        # numerical values i.e. limedata[rw, ind] is index of
                        # this array. value is string val
                        new_feature = feature + '=' + self.s_categorical_names[ind][int(limedata[rw, ind])]
                        # categorical feature:
                        aif360data[rw, self.s_feature_names_with_one_hot_encoding.index(new_feature)] = 1.0
                else: # just copy value
                    aif360data[rw, self.s_feature_names_with_one_hot_encoding.index(feature)] = limedata[rw, ind]

        return aif360data
