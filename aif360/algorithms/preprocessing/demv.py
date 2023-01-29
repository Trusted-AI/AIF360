import numpy as np
import pandas as pd

from aif360.algorithms import Transformer
from aif360.datasets import StructuredDataset


def _balance_set(w_exp, w_obs, df: pd.DataFrame, tot_df, round_level=None, debug=False, k=-1):
    disp = round(w_exp / w_obs, round_level) if round_level else w_exp / w_obs
    disparity = [disp]
    i = 0
    while disp != 1 and i != k:
        if w_exp / w_obs > 1:
            df = df.append(df.sample())
        elif w_exp / w_obs < 1:
            df = df.drop(df.sample().index, axis=0)
        w_obs = len(df) / len(tot_df)
        disp = round(
            w_exp / w_obs, round_level) if round_level else w_exp / w_obs
        disparity.append(disp)
        if debug:
            print(w_exp / w_obs)
        i += 1
    return df, disparity, i


def _sample(d: pd.DataFrame, s_vars: list, label: str, round_level: float, debug: bool = False,
            i: int = 0, G=None, cond: bool = True, stop=-1):
    if G is None:
        G = []
    d = d.copy()
    n = len(s_vars)
    disparities = []
    iter = 0
    if i == n:
        for l in np.unique(d[label]):
            g = d[(cond) & (d[label] == l)]
            if len(g) > 0:
                w_exp = (len(d[cond]) / len(d)) * \
                    (len(d[d[label] == l]) / len(d))
                w_obs = len(g) / len(d)
                g_new, disp, k = _balance_set(
                    w_exp, w_obs, g, d, round_level, debug, stop)
                g_new = g_new.astype(g.dtypes.to_dict())
                disparities.append(disp)
                G.append(g_new)
                iter = max(iter, k)
        return G, iter, disparities
    else:
        s = s_vars[i]
        i = i + 1
        G1, k1, disp1 = _sample(d, s_vars, label, round_level, debug, i,
                                G.copy(), cond=cond & (d[s] == 0), stop=stop)
        G2, k2, disp2 = _sample(d, s_vars, label, round_level, debug, i,
                                G.copy(), cond=cond & (d[s] == 1), stop=stop)
        G += G1
        G += G2
        iter = max([iter, k1, k2])
        new_disps = disp1 + disp2
        disparities.append(new_disps)
        limit = 1
        for s in s_vars:
            limit *= len(np.unique(d[s]))
        if len(G) == limit * len(np.unique(d[label])):
            return pd.DataFrame(G.pop().append([g for g in G]).sample(frac=1, random_state=2)), disparities, iter
        else:
            return G, iter, disparities


class DEMV(Transformer):
    """
    Debiaser for Multiple Variables(DEMV) is a pre-processing algorithm for binary
    and multi-class datasets that mitigates bias by perfectly balancing the sensitive groups
    identified by each possible sensitive variables' value and each label's value [1].

    References:
         [1] G. d'Aloisio, A. D'Angelo, A. Di Marco, e G. Stilo,
         «Debiaser for Multiple Variables to enhance fairness in classification tasks»,
         Information Processing & Management, vol. 60, mar. 2023, doi: 10.1016/j.ipm.2022.103226.

    Based on the code from: https://github.com/giordanoDaloisio/demv
    """

    def __init__(self, round_level=1, debug=False):
        """
        Parameters
        ----------
        round_level : float, optional
            Tolerance value to balance the sensitive groups (default is None)
        debug : bool, optional
            Prints w_exp/w_obs, useful for debugging (default is False)
        """
        self.disparities = []
        self.round_level = round_level
        self.debug = debug
        self.iter = 0
        super(DEMV, self).__init__()

    def fit(self, dataset: StructuredDataset):
        """
        Balances the dataset's sensitive groups

        Args
        ----------
        dataset : StructuredDataset
            Dataset to be balanced

        Returns
        -------
         StructuredDataset:
            Balanced dataset
        """
        return self.fit_transform(dataset)

    def transform(self, dataset: StructuredDataset):
        """
        Balances the dataset's sensitive groups

        Args
        ----------
        dataset : StructuredDataset
            Dataset to be balanced

        Returns
        -------
         StructuredDataset:
            Balanced dataset
        """
        return self.fit_transform(dataset)

    def fit_transform(self, dataset: StructuredDataset):
        """
        Balances the dataset's sensitive groups

        Args
        ----------
        dataset : StructuredDataset
            Dataset to be balanced

        Returns
        -------
         StructuredDataset:
            Balanced dataset
        """
        protected_attrs = dataset.protected_attribute_names
        label_name = dataset.label_names[0]
        df, _ = dataset.convert_to_dataframe()
        df_new, disparities, iters = _sample(df, protected_attrs,
                                             label_name, self.round_level,
                                             self.debug, 0, [], True)
        self.iter = iters
        self.disparities = disparities
        new_data = StructuredDataset(df_new, label_names=dataset.label_names,
                                     protected_attribute_names=dataset.protected_attribute_names,
                                     unprivileged_protected_attributes=dataset.unprivileged_protected_attributes,
                                     privileged_protected_attributes=dataset.privileged_protected_attributes)
        return new_data

    def get_iters(self):
        """
        Gets the maximum number of iterations

        Returns
        -------
        int:
            maximum number of iterations
        """
        return self.iter

    def get_disparities(self):
        """
        Returns the list of w_exp/w_obs

        Returns
        -------
        list:
            list of disparities values
        """
        return np.array(self.disparities)
