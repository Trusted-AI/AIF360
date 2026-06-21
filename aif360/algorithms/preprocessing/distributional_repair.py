import numpy as np
import pandas as pd

from aif360.algorithms import Transformer
from aif360.datasets import StandardDataset

from sklearn.neighbors import KernelDensity
import ot


class DistributionalRepair(Transformer):
    """Distributional Repair class for mitigating bias in datasets.

    Adapted from the work of Abigail Longbridge et al.
    https://arxiv.org/pdf/2403.13864

    This class implements the Distributional Repair algorithm to mitigate bias
    in datasets by aligning the distributions of protected and unprotected groups
    for each feature conditioned on the outcome variable.

    """

    def __init__(self, s, u, x, y, continuous_features, n_q=250):
        """
        Initialize the Distributional Repair transformer.

        Args:
        s (str): Name of the protected attribute.
        u (str): Name of the unprotected variable.
        x (list): List of feature names to be repaired (remaining observations).
        y (str): Name of the outcome variable.
        continuous_features (list): List of continuous feature names.
        n_q (int, optional): Number of probability function supports. Defaults to 250.
        """
        super(DistributionalRepair, self).__init__()
        self.s = s
        self.u = u
        self.x = x
        self.y = y
        self.n_q = n_q
        self.continuous_features = continuous_features

    def fit(self, dataset_R):
        """Fit the Distributional Repair transformer to create an optimal transport plan from the reference dataset.

        Args:
            dataset_R (StandardDataset): Dataset to fit the transformer.

        Returns:
            self: Fitted transformer.
        """
        dataframe_R = dataset_R.convert_to_dataframe()[0]
        self.supports = {}
        self.pmf_0s = {}
        self.pmf_1s = {}
        self.T_0s = {}
        self.T_1s = {}
        self.s_R, self.u_R, self.x_R, self.y_R = self._split_dataframe(dataframe_R)
        for feat in self.x:
            continuous = feat in self.continuous_features
            for u_val in self.u_R.unique():
                if continuous:
                    self._continuous_fit(feat, u_val)
                else:
                    self._discrete_fit(feat, u_val)
        return self

    def _continuous_fit(self, feat, u_val):
        support = self._get_support(feat, u_val)
        self.supports[(feat, u_val)] = support
        pmf_0, pmf_1 = self._get_pmfs(feat, u_val)
        barycenter = self._get_barycenter(pmf_0, pmf_1, feat, u_val)
        T_0, T_1 = self._get_transport_plans(pmf_0, pmf_1, barycenter, feat, u_val)
        self.pmf_0s[(feat, u_val)] = pmf_0
        self.pmf_1s[(feat, u_val)] = pmf_1
        self.T_0s[(feat, u_val)] = T_0
        self.T_1s[(feat, u_val)] = T_1

    def _discrete_fit(self, feat, u_val):
        if self._is_valid_data(u_val):
            pmf_0, pmf_1 = self._get_discrete_pmfs(feat, u_val)
            T = self._get_discrete_transport_plan(pmf_0, pmf_1)
            self.pmf_0s[(feat, u_val)] = pmf_0
            self.pmf_1s[(feat, u_val)] = pmf_1
            self.T_0s[(feat, u_val)] = T
        else:
            self.pmf_0s[(feat, u_val)] = None
            self.pmf_1s[(feat, u_val)] = None

    def transform(self, dataset_D):
        """Transform the dataset to apply the OT plan.

        Args:
            dataset_D (StandardDataset): Dataset to be transformed.

        Returns:
            StandardDataset: Transformed dataset with bias mitigated.
        """
        dataframe_D = dataset_D.convert_to_dataframe()[0]
        s_D, u_D, x_D, y_D = self._split_dataframe(dataframe_D)
        tilde_x_D = x_D.copy()
        for feat in self.x:
            continuous = feat in self.continuous_features
            for u_val in u_D.unique():
                if continuous:
                    support = self.supports[(feat, u_val)]
                    T_0 = self.T_0s[(feat, u_val)]
                    T_1 = self.T_1s[(feat, u_val)]
                    tilde_x_D = self._continuous_transform(s_D, u_D, x_D, feat, tilde_x_D, u_val, support, T_0, T_1)
                else:
                    tilde_x_D = self._discrete_transform(s_D, u_D, x_D, feat, tilde_x_D, u_val)
        
        tilde_dataframe_D = pd.concat([tilde_x_D, dataframe_D.drop(columns=self.x)], axis=1)
        tilde_dataset_D = StandardDataset(df=tilde_dataframe_D, 
                                          label_name=self.y,
                                          favorable_classes=[1],
                                          protected_attribute_names=[self.s],
                                          privileged_classes=[[1]])
        return tilde_dataset_D

    def fit_transform(self, dataframe_R, dataframe_A):
        """Fit and transform the datasets.

        Args:
            dataframe_R (DataFrame): Reference dataset.
            dataframe_A (DataFrame): Dataset to be transformed.

        Returns:
            tuple: Transformed reference dataset and transformed dataset.
        """
        self.fit(dataframe_R)
        tilde_dataframe_A = self.transform(dataframe_A)
        tilde_dataframe_R = self.transform(dataframe_R)
        return tilde_dataframe_R, tilde_dataframe_A

    def _split_dataframe(self, dataframe):
        s_D = dataframe[self.s]
        u_D = dataframe[self.u]
        x_D = dataframe[self.x]
        y_D = dataframe[self.y]
        return s_D, u_D, x_D, y_D

    def _get_support(self, feat, u_val):
        min_val = np.min(self.x_R[(self.u_R == u_val)][feat]) - np.ptp(self.x_R[(self.u_R == u_val)][feat])*0.1
        max_val = np.max(self.x_R[(self.u_R == u_val)][feat]) + np.ptp(self.x_R[(self.u_R == u_val)][feat])*0.1
        return np.linspace(min_val, max_val, self.n_q).reshape(-1, 1)

    def _get_pmfs(self, feat, u_val):
        kde_0 = KernelDensity(kernel='gaussian', bandwidth='silverman').fit(self.x_R[(self.u_R == u_val) & (self.s_R == 0.0)][feat].values.reshape(-1, 1))
        pmf_0 = np.exp(kde_0.score_samples(self.supports[(feat, u_val)]))
        kde_1 = KernelDensity(kernel='gaussian', bandwidth='silverman').fit(self.x_R[(self.u_R == u_val) & (self.s_R == 1.0)][feat].values.reshape(-1, 1))
        pmf_1 = np.exp(kde_1.score_samples(self.supports[(feat, u_val)]))
        pmf_0 /= np.sum(pmf_0)
        pmf_1 /= np.sum(pmf_1)
        if np.any(np.isnan(pmf_0)) or np.any(np.isnan(pmf_1)):
            raise ZeroDivisionError("One or more PMFs have sum zero")
        return pmf_0, pmf_1

    def _get_barycenter(self, pmf_0, pmf_1, feat, u_val):
        M = ot.utils.dist(self.supports[(feat, u_val)], self.supports[(feat, u_val)])
        A = np.vstack([pmf_0, pmf_1]).T
        barycenter = ot.bregman.barycenter(A, M, 10)
        if np.any(np.isnan(pmf_0)) or np.any(np.isnan(pmf_1)):
            raise RuntimeError("No valid barycenter was found, try to increase reg")
        return barycenter

    def _get_transport_plans(self, pmf_0, pmf_1, barycenter, feat, u_val):
        M = ot.utils.dist(self.supports[(feat, u_val)], self.supports[(feat, u_val)])
        T_0 = ot.emd(pmf_0, barycenter, M)
        T_1 = ot.emd(pmf_1, barycenter, M)
        return T_0, T_1

    def _is_valid_data(self, u_val):
        return (len(self.x_R[(self.u_R == u_val) & (self.s_R == 0)]) > 1) and (len(self.x_R[(self.u_R == u_val) & (self.s_R == 1)]) > 1)

    def _get_discrete_pmfs(self, feat, u_val):
        pmf_0 = self.x_R[(self.u_R == u_val) & (self.s_R == 0)][feat].value_counts()
        pmf_1 = self.x_R[(self.u_R == u_val) & (self.s_R == 1)][feat].value_counts()
        return pmf_0, pmf_1

    def _get_discrete_transport_plan(self, pmf_0, pmf_1):
        M = ot.dist(pmf_0.index.values.reshape(-1, 1), pmf_1.index.values.reshape(-1, 1))
        weights = [pmf_0.values / pmf_0.values.sum(), pmf_1.values / pmf_1.values.sum()]
        return ot.emd(weights[0], weights[1], M)

    def _continuous_transform(self, s_D, u_D, x_D, feat, tilde_x_D, u_val, support, T_0, T_1):
        for i, row in x_D[(u_D == u_val)].iterrows():
            if s_D[i] == 1:
                tilde_x_D.loc[i, feat] = self._repair_data(row[feat], support[:, 0], support[:, 0], T_1)
            else:
                tilde_x_D.loc[i, feat] = self._repair_data(row[feat], support[:, 0], support[:, 0], T_0)
        return tilde_x_D

    def _discrete_transform(self, s_D, u_D, x_D, feat, tilde_x_D, u_val):
        pmf_0 = self.pmf_0s[(feat, u_val)]
        pmf_1 = self.pmf_1s[(feat, u_val)]
        T = self.T_0s[(feat, u_val)]
        
        if pmf_0 is None or pmf_1 is None:
            return tilde_x_D
        
        for i, row in x_D[(u_D == u_val)].iterrows():
            if s_D[i] == 1:
                tilde_x_D.loc[i, feat] = self._repair_data(row[feat], pmf_1.index.values, pmf_0.index.values, T.T, i_split=False, j_split=False)
            else:
                tilde_x_D.loc[i, feat] = self._repair_data(row[feat], pmf_0.index.values, pmf_1.index.values, T, i_split=False, j_split=False)
        return tilde_x_D

    def _repair_data(self, x, support_i, support_j, T, i_split=True, j_split=False):
        if i_split:
            idx = np.searchsorted(support_i, x, side='left')
            if idx == 0 or idx == len(support_i):
                i = min(idx, len(support_i)-1)
            else:
                interp = float(x - support_i[idx-1]) / np.diff(support_i)[0]
                if np.round(interp, 4) == 1.0:
                    i = idx
                else:
                    i = np.random.choice([idx-1, idx], p=[1-interp, interp])
        else:
            i_indices = np.argwhere(support_i == x)
            if len(i_indices) > 0:
                i = i_indices[0,0]
            else:
                i = np.argmin(np.abs(support_i - x))

        if not j_split:
            if np.sum(T[i]) > 0.0:
                j = np.random.choice(T.shape[1], p=(T[i] / np.sum(T[i]))) # stochastic choice of which marginal entry to transport to
            else:
                j = i
            x_repaired = support_j[j]
        else:
            row = T[i] / np.sum(T[i])
            x_repaired = 0.5*x + 0.5*row@support_j
        return x_repaired