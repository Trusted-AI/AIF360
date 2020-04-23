import numpy as np

from aif360.datasets import StructuredDataset
'''
Multiclass supports the multiple values in the favorable and unfavorable label's
'''
class MulticlassLabelDataset(StructuredDataset):
    """Base class for all structured datasets with multiclass labels."""

    def __init__(self, favorable_label = [1.], unfavorable_label = [0.], **kwargs):
        """
        Args:
            favorable_label (list): Label value which is considered favorable
                (i.e. "positive").
            unfavorable_label (list): Label value which is considered
                unfavorable (i.e. "negative").
            **kwargs: StructuredDataset arguments.
        """
        self.favorable_label = favorable_label
        self.unfavorable_label = unfavorable_label

        super(MulticlassLabelDataset, self).__init__(**kwargs)
        

    def validate_dataset(self):
        """Error checking and type validation.
​
        Raises:
            ValueError: `labels` must be shape [n, 1].
            ValueError: `favorable_label` and `unfavorable_label` must be the
                only values present in `labels`.
        """
        # fix scores before validating
        if np.all(self.scores == self.labels):
            for i in range(0,len(self.scores)):
                if self.scores[i] in self.favorable_label:
                    self.scores[i] = float(1)
                else:
                    self.scores[i] = float(0)

        super(MulticlassLabelDataset, self).validate_dataset()

        # =========================== SHAPE CHECKING ===========================
        # Verify if the labels are only 1 column
        if self.labels.shape[1] != 1:
            raise ValueError("MulticlassLabelDataset only supports single-column "
                "labels:\n\tlabels.shape = {}".format(self.labels.shape))

        # =========================== VALUE CHECKING ===========================
        # Check if the favorable and unfavorable labels match those in the dataset
        if (not set(self.labels.ravel()) <=
                set(self.favorable_label + (self.unfavorable_label))):
            raise ValueError("The favorable and unfavorable labels provided do "
                             "not match the labels in the dataset.")
