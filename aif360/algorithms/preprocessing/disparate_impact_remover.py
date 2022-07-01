import numpy as np

from aif360.algorithms import Transformer


class DisparateImpactRemover(Transformer):
    """Disparate impact remover is a preprocessing technique that edits feature
    values increase group fairness while preserving rank-ordering within groups
    [1]_.

    References:
        .. [1] M. Feldman, S. A. Friedler, J. Moeller, C. Scheidegger, and
           S. Venkatasubramanian, "Certifying and removing disparate impact."
           ACM SIGKDD International Conference on Knowledge Discovery and Data
           Mining, 2015.
    """

    def __init__(self, repair_level=1.0, sensitive_attribute=''):
        """
        Args:
            repair_level (float): Repair amount. 0.0 is no repair while 1.0 is
                full repair.
            sensitive_attribute (str): Single protected attribute with which to
                do repair.
        """
        super(DisparateImpactRemover, self).__init__(repair_level=repair_level)
        # avoid importing early since this package can throw warnings in some
        # jupyter notebooks
        from BlackBoxAuditing.repairers.GeneralRepairer import Repairer
        self.Repairer = Repairer

        if not 0.0 <= repair_level <= 1.0:
            raise ValueError("'repair_level' must be between 0.0 and 1.0.")
        self.repair_level = repair_level

        self.sensitive_attribute = sensitive_attribute

    def fit_transform(self, dataset):
        """Run a repairer on the non-protected features and return the
        transformed dataset.

        Args:
            dataset (BinaryLabelDataset): Dataset that needs repair.
        Returns:
            dataset (BinaryLabelDataset): Transformed Dataset.

        Note:
            In order to transform test data in the same manner as training data,
            the distributions of attributes conditioned on the protected
            attribute must be the same.
        """
        if not self.sensitive_attribute:
            self.sensitive_attribute = dataset.protected_attribute_names[0]

        features = dataset.features.tolist()
        index = dataset.feature_names.index(self.sensitive_attribute)
        repairer = self.Repairer(features, index, self.repair_level, False)

        repaired = dataset.copy()
        repaired_features = repairer.repair(features)
        repaired.features = np.array(repaired_features, dtype=np.float64)
        # protected attribute shouldn't change
        repaired.features[:, index] = repaired.protected_attributes[:, repaired.protected_attribute_names.index(self.sensitive_attribute)]

        return repaired
