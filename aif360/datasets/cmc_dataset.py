import os.path
import pandas as pd
from aif360.datasets import StructuredDataset


class CmcDataset(StructuredDataset):
    """Contraceptive Method Choice Dataset.

    See :file:`aif360/data/raw/cmc/README.md`.
    """

    def __init__(self, label_name=['contr_use'],
                 favorable_classes=[2.0],
                 protected_attribute_names=['wife_religion', 'wife_work'],
                 privileged_classes=[[1], [1]],
                 instance_weights_name=None):
        """This is a multi-class dataset, meaning that the label has more than 2 values.
        See :obj:`StructuredDataset` for a description of the arguments.

        Additional Parameters
        --------------------
        favorable_classes: Positive label of the dataset

        """
        self.favorable_class = favorable_classes

        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw', 'cmc', 'cmc.data')
        col_names = ['wife_age', 'wife_edu', 'hus_edu', 'num_child', 'wife_religion', 'wife_work', 'hus_occ', 'living',
                     'media', 'contr_use']
        try:
            data = pd.read_csv(data_path, names=col_names)
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following file:")
            print("\n\thttps://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data")
            print("\nand place the file, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
                os.path.abspath(__file__), '..', '..', 'data', 'raw', 'cmc'))))
            import sys
            sys.exit(1)
        super(CmcDataset, self).__init__(df=data, label_names=label_name,
                                         protected_attribute_names=protected_attribute_names,
                                         instance_weights_name=instance_weights_name,
                                         privileged_protected_attributes=privileged_classes)
