import numpy as np
import pandas as pd

from aif360.algorithms import Transformer
from aif360.algorithms.preprocessing.demv_helpers import sample
from aif360.datasets import StructuredDataset
from aif360.datasets import BinaryLabelDataset
from aif360.datasets.multiclass_label_dataset import MulticlassLabelDataset

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

    def __init__(self, round_level=1, stop=10000, debug=False):
        """
        Parameters
        ----------
        round_level : float, optional
            Tolerance value to balance the sensitive groups (default is 1)
        stop : int, optional
            Maximum number of iterations to balance the sensitive groups (default is 10000)
        debug : bool, optional
            Prints w_exp/w_obs, useful for debugging (default is False)
        """
        self.disparities = []
        self.round_level = round_level
        self.stop = stop
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
         StructuredDataset: Balanced dataset
        """
        protected_attrs = dataset.protected_attribute_names
        label_name = dataset.label_names[0]
        unpriv_prot_attr = dataset.unprivileged_protected_attributes[0]
        priv_prot_attr = dataset.privileged_protected_attributes[0]

        df, _ = dataset.convert_to_dataframe()
        df_new, disparities, iters = sample(df, protected_attrs,
                                             label_name, self.round_level,
                                             self.debug, 0, [], True, stop=self.stop)
        self.iter = iters
        self.disparities = disparities
        if len(df_new[label_name].unique()) == 2:
            if dataset.favorable_label and dataset.unfavorable_label:
                new_data = BinaryLabelDataset(favorable_label=dataset.favorable_label,
                unfavorable_label=dataset.unfavorable_label,
                df=df_new,
                label_names=[label_name],
                protected_attribute_names=protected_attrs,
                unprivileged_protected_attributes=unpriv_prot_attr,privileged_protected_attributes=priv_prot_attr)
            elif dataset.favorable_label:
                new_data = BinaryLabelDataset(
                favorable_label=dataset.favorable_label,
                df=df_new,
                label_names=[label_name],
                protected_attribute_names=protected_attrs,
                unprivileged_protected_attributes=unpriv_prot_attr,privileged_protected_attributes=priv_prot_attr)
            elif dataset.unfavorable_label:
                new_data = BinaryLabelDataset(
                unfavorable_label=dataset.unfavorable_label,
                df=df_new,
                label_names=[label_name],
                protected_attribute_names=protected_attrs,
                unprivileged_protected_attributes=unpriv_prot_attr,privileged_protected_attributes=priv_prot_attr)
            else:
                new_data = BinaryLabelDataset(
                df=df_new,
                label_names=[label_name],
                protected_attribute_names=protected_attrs,
                unprivileged_protected_attributes=unpriv_prot_attr,privileged_protected_attributes=priv_prot_attr)
        else:
            if dataset.favorable_label and dataset.unfavorable_label:
                new_data = MulticlassLabelDataset(
                    favorable_label=dataset.favorable_label, 
                    unfavorable_label=dataset.unfavorable_label,
                    df=df_new, 
                    label_names=[label_name],
                    protected_attribute_names=protected_attrs,
                    unprivileged_protected_attributes=unpriv_prot_attr,privileged_protected_attributes=priv_prot_attr)
            elif dataset.favorable_label:
                new_data = MulticlassLabelDataset(
                favorable_label=dataset.favorable_label, 
                df=df_new, 
                label_names=[label_name],
                protected_attribute_names=protected_attrs,
                unprivileged_protected_attributes=unpriv_prot_attr,privileged_protected_attributes=priv_prot_attr)
            elif dataset.unfavorable_label:
                new_data = MulticlassLabelDataset(
                unfavorable_label=dataset.unfavorable_label,
                df=df_new, 
                label_names=[label_name],
                protected_attribute_names=protected_attrs,
                unprivileged_protected_attributes=unpriv_prot_attr,privileged_protected_attributes=priv_prot_attr)
            else:
                new_data = MulticlassLabelDataset(
                    df=df_new, 
                    label_names=[label_name],
                    protected_attribute_names=protected_attrs,
                    unprivileged_protected_attributes=unpriv_prot_attr,privileged_protected_attributes=priv_prot_attr)
        return new_data

    def get_iters(self):
        """
        Gets the maximum number of iterations

        Returns:
        int: maximum number of iterations
        """
        return self.iter

    def get_disparities(self):
        """
        Returns the list of w_exp/w_obs

        Returns:
        list: list of disparities values
        """
        return np.array(self.disparities)
