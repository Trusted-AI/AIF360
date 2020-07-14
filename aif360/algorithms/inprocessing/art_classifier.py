import numpy as np

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms import Transformer


class ARTClassifier(Transformer):
    """Wraps an instance of an :obj:`art.classifiers.Classifier` to extend
    :obj:`~aif360.algorithms.Transformer`.
    """

    def __init__(self, art_classifier):
        """Initialize ARTClassifier.

        Args:
            art_classifier (art.classifier.Classifier): A Classifier
                object from the `adversarial-robustness-toolbox`_.

        .. _adversarial-robustness-toolbox:
           https://github.com/Trusted-AI/adversarial-robustness-toolbox
        """
        super(ARTClassifier, self).__init__(art_classifier=art_classifier)
        self._art_classifier = art_classifier

    def fit(self, dataset, batch_size=128, nb_epochs=20):
        """Train a classifer on the input.

        Args:
            dataset (Dataset): Training dataset.
            batch_size (int): Size of batches (passed through to ART).
            nb_epochs (int): Number of epochs to use for training (passed
                through to ART).

        Returns:
            ARTClassifier: Returns self.
        """
        self._art_classifier.fit(dataset.features, dataset.labels,
                                 batch_size=batch_size, nb_epochs=nb_epochs)
        return self

    def predict(self, dataset, logits=False):
        """Perform prediction for the input.

        Args:
            dataset (Dataset): Test dataset.
            logits (bool, optional): True is prediction should be done at the
                logits layer (passed through to ART).

        Returns:
            Dataset: Dataset with predicted labels in the `labels` field.
        """
        pred_labels = self._art_classifier.predict(dataset.features,
            dataset.labels, logits=logits)

        if isinstance(dataset, BinaryLabelDataset):
            pred_labels = np.argmax(pred_labels, axis=1).reshape((-1, 1))

        pred_dataset = dataset.copy()
        pred_dataset.labels = pred_labels

        return pred_dataset
