import os.path as osp

import pandas as pd
import numpy as np
try:
    from rpy2 import robjects
    from rpy2.robjects.vectors import StrVector
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
except ImportError as error:
    from logging import warning
    warning("{}: FairAdapt will be unavailable. To install, run:\n"
            "pip install 'aif360[FairAdapt]'".format(error))
from sklearn.base import BaseEstimator

from aif360.sklearn.utils import check_inputs, check_groups


class FairAdapt(BaseEstimator):
    """Fair Data Adaptation.

    Fairadapt is a pre-processing technique that can be used for both fair
    classification and fair regression [#plecko20]_. The method is a causal
    inference approach to bias removal and it relies on the causal graph for
    the dataset. The original implementation is in R [#plecko21]_.

    References:
        .. [#plecko20] `D. Plečko and N. Meinshausen,
           "Fair Data Adaptation with Quantile Preservation,"
           Journal of Machine Learning Research, 2020.
           <https://www.jmlr.org/papers/volume21/19-966/19-966.pdf>`_
        .. [#plecko21] `D. Plečko and N. Bennett and N. Meinshausen,
           "FairAdapt: Causal Reasoning for Fair Data Pre-processing,"
           arXiv, 2021. <https://arxiv.org/abs/2110.10200>`_

    Attributes:
        prot_attr_ (str or list(str)): Protected attribute(s) used for fair data
            adaptation.
        groups_ (array, shape (n_groups,)): A list of group labels known to the
            transformer.
    """

    def __init__(self, prot_attr, adj_mat):
        """
        Args:
            prot_attr (single label): Name of the protected attribute. Must be
                binary.
            adj_mat (array-like): A 2-dimensional array representing the
                adjacency matrix of the causal diagram of the data generating
                process. Row/column order must match `X_train`.
        """
        self.prot_attr = prot_attr
        self.adj_mat = adj_mat

        # R packages need to run FairAdapt
        pkgs = ('ranger', 'fairadapt')
        # selectively install the missing packages
        pkgs = [p for p in pkgs if not robjects.packages.isinstalled(p)]
        if len(pkgs) > 0:
            utls = robjects.packages.importr('utils')
            utls.chooseCRANmirror(ind=1)
            utls.install_packages(StrVector(pkgs))

    def fit_transform(self, X_train, y_train, X_test):
        """Remove bias from the given dataset by fair adaptation.

        Args:
            X_train (pandas.DataFrame): Training data frame (including the
                protected attribute).
            y_train (pandas.Series): Training labels.
            X_test (pandas.DataFrame): Test data frame (including the protected
                attribute).

        Returns:
            tuple:
                Transformed inputs.

                * **X_fair_train** (pandas.DataFrame) -- Transformed training
                  data.
                * **y_fair_train** (array-like) -- Transformed training labels.
                * **X_fair_test** (pandas.DataFrame) -- Transformed test data.

        """
        # merge X_train and y_train
        df_train = pd.concat([X_train, y_train], axis=1)
        groups, self.prot_attr_ = check_groups(X_train, self.prot_attr, ensure_binary=True)
        self.groups_ = np.unique(groups)

        wrapper = osp.join(osp.dirname(osp.abspath(__file__)), 'fairadapt.R')
        robjects.r.source(wrapper)
        FairAdapt_R = robjects.r['wrapper']
        # convert to Pandas with a local converter
        with localconverter(robjects.default_converter + pandas2ri.converter):
            train_data = robjects.conversion.py2rpy(df_train)
            test_data = robjects.conversion.py2rpy(X_test)
            adj_mat = robjects.conversion.py2rpy(self.adj_mat)

        # run FairAdapt in R
        res = FairAdapt_R(
            train_data=train_data,
            test_data=test_data,
            adj_mat=adj_mat,
            prot_attr=self.prot_attr_,
            outcome=y_train.name
        )

        with localconverter(robjects.default_converter + pandas2ri.converter):
            X_fair_train = robjects.conversion.rpy2py(res.rx2('train'))
            X_fair_test = robjects.conversion.rpy2py(res.rx2('test'))
        X_fair_train.columns = [y_train.name] + X_train.columns.tolist()
        y_fair_train = X_fair_train.pop(y_train.name)
        X_fair_test.columns = X_test.columns

        return X_fair_train, y_fair_train, X_fair_test
