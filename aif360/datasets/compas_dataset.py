import os

import pandas as pd

from aif360.datasets import StandardDataset


default_mappings = {
    'label_maps': [{1.0: 'Did recid.', 0.0: 'No recid.'}],
    'protected_attribute_maps': [{0.0: 'Male', 1.0: 'Female'},
                                 {1.0: 'Caucasian', 0.0: 'Not Caucasian'}]
}

def default_preprocessing(df):
    """Perform the same preprocessing as the original analysis:
    https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    """
    return df[(df.days_b_screening_arrest <= 30)
            & (df.days_b_screening_arrest >= -30)
            & (df.is_recid != -1)
            & (df.c_charge_degree != 'O')
            & (df.score_text != 'N/A')]

class CompasDataset(StandardDataset):
    """ProPublica COMPAS Dataset.

    See :file:`aif360/data/raw/compas/README.md`.
    """

    def __init__(self, label_name='two_year_recid', favorable_classes=[0],
                 protected_attribute_names=['sex', 'race'],
                 privileged_classes=[['Female'], ['Caucasian']],
                 instance_weights_name=None,
                 categorical_features=['age_cat', 'c_charge_degree',
                     'c_charge_desc'],
                 features_to_keep=['sex', 'age', 'age_cat', 'race',
                     'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                     'priors_count', 'c_charge_degree', 'c_charge_desc',
                     'two_year_recid'],
                 features_to_drop=[], na_values=[],
                 custom_preprocessing=default_preprocessing,
                 metadata=default_mappings):
        """See :obj:`StandardDataset` for a description of the arguments.

        Note: The label value 0 in this case is considered favorable (no
        recidivism).

        Examples:
            In some cases, it may be useful to keep track of a mapping from
            `float -> str` for protected attributes and/or labels. If our use
            case differs from the default, we can modify the mapping stored in
            `metadata`:

            >>> label_map = {1.0: 'Did recid.', 0.0: 'No recid.'}
            >>> protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
            >>> cd = CompasDataset(protected_attribute_names=['sex'],
            ... privileged_classes=[['Male']], metadata={'label_map': label_map,
            ... 'protected_attribute_maps': protected_attribute_maps})

            Now this information will stay attached to the dataset and can be
            used for more descriptive visualizations.
        """

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '..', 'data', 'raw', 'compas', 'compas-scores-two-years.csv')

        try:
            df = pd.read_csv(filepath, index_col='id', na_values=na_values)
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following file:")
            print("\n\thttps://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv")
            print("\nand place it, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', '..', 'data', 'raw', 'compas'))))
            import sys
            sys.exit(1)

        super(CompasDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
