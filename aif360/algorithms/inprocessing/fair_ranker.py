import numpy as np
try:
    import fairsearchcore as fsc
except ImportError as error:
    print("Import error: %s" % (error))

from aif360.algorithms import Transformer
from aif360.metrics.utils import compute_boolean_conditioning_vector

class FairRanker(Transformer):
    def __init__(self, k, p, alpha, unprivileged_groups, privileged_groups,
                 label_name=None):
        """
        Args:
            k (int): Number of topK elements returned (value should be between
                10 and 400)
            p (float): Proportion of protected candidates in the topK elements
                (value should be between 0.02 and 0.98)
            alpha (float): Significance level (value should be between 0.01 and
                0.15)
            label_name (str): Single label to use as a score for the ranker.
        """
        # TODO: make only calculate single m -- not whole table
        self.fair_ranker = fsc.Fair(k, p, alpha)
        self.unprivileged_groups = uprivileged_groups
        self.privileged_groups = privileged_groups
        self.label_name = label_name

    # fit_rerank?
    def fit_transform(self, dataset):
        try:
            label_index = dataset.label_names.index(self.label_name)
        except ValueError:
            raise ValueError("{} not in 'dataset.label_names':\n\t{}".format(
                    self.label_name, dataset.label_names))

        # unpr = compute_boolean_conditioning_vector(dataset.protected_attributes,
        #         dataset.protected_attribute_names, self.unprivileged_groups)
        priv = compute_boolean_conditioning_vector(dataset.protected_attributes,
                dataset.protected_attribute_names, self.privileged_groups)

        def iter_fairscoredoc():
            for i in range(len(dataset.labels)):
                yield fsc.model.FairScoreDoc(id=i,
                        score=dataset.labels[i, label_index],
                        is_protected=priv[i])

        ranking = self.fair_ranker.re_rank(iter_fairscoredoc())
        ranked_order = [fsd.id for fsd in ranking]

        reranked_dataset = dataset.copy(True)
        # TODO: make RankedDataset class
        reranked_dataset.features = reranked_dataset.features[ranked_order]
        reranked_dataset.labels = reranked_dataset.labels[ranked_order]
        reranked_dataset.scores = reranked_dataset.scores[ranked_order]
        reranked_dataset.protected_attributes = reranked_dataset.protected_attributes[ranked_order]
        reranked_dataset.instance_names = reranked_dataset.instance_names[ranked_order]
        reranked_dataset.instance_weights = reranked_dataset.instance_weights[ranked_order]
        return reranked_dataset
