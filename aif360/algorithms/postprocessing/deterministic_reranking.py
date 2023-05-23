import numpy as np
import pandas as pd

from aif360.algorithms import Transformer
from aif360.datasets import StructuredDataset, RegressionDataset

class DeterministicReranking(Transformer):
    """A collection of algorithms for construction of fair ranked candidate lists. [1]_ .

    References:
        .. [1] KDD '19: Proceedings of the 25th ACM SIGKDD 
           International Conference on Knowledge Discovery & Data Mining, July 2019, Pages 2221-2231.
    """

    def __init__(self,
            unprivileged_groups: list[dict],
            privileged_groups: list[dict]):
        """
        Args:
            unprivileged_groups (list(dict)): Representation for the unprivileged
                group.
            privileged_groups (list(dict)): Representation for the privileged
                group.
        """
        if len(unprivileged_groups) != 1 or len(privileged_groups) != 1:
            raise ValueError("Only one unprivileged group or privileged group supported.")
        if list(unprivileged_groups[0].keys())[0] != list(privileged_groups[0].keys())[0]:
            raise ValueError("Different sensitive attributes (not values) specified for unprivileged and privileged groups.")      

        super(DeterministicReranking, self).__init__(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)
        
        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups


    def fit_predict(self,
                    dataset: RegressionDataset,
                    rec_size: int,
                    target_prop: dict,
                    rerank_type: str='Constrained',
                    normalize_scores: bool=False
                    ):
        """Construct a ranking of candidates in the dataset according to specified proportions of groups.

        Args:
            dataset (RegressionDataset): Dataset to rerank.
            rec_size (int): Number of candidates in the output.
            target_prop (dict): Desired proportion of groups in the output. Format is \
                {protected_attribute_value: proportion}.
            rerank_type: Greedy, Conservative, Relaxed, or Constrained. Determines the type of algorithm \
                as described in the original paper.
            normalize_scores: normalize label (score) values in the resulting ranking. If True, uses the default \
                behavior of RegressionDataset.

        Returns:
            RegressionDataset: The reranked dataset.
        """
        
        s_vals = set(self.unprivileged_groups[0].values()).union(set(self.privileged_groups[0].values()))

        if rec_size <= 0:
            raise ValueError(f"Output size should be greater than 0, got {rec_size}.")
        if np.any(set(target_prop.keys()) != set(s_vals)):
            raise ValueError("""Proportion specifications do not match. \
                `target_prop` should have sensitive attribute values as keys.""")

        if len(dataset.label_names) != 1:
            raise ValueError(f"Dataset must have exactly one label, got {len(dataset.label_names)}.")
        
        items = dataset.convert_to_dataframe()[0]
        items = items.sort_values(axis=0, by=dataset.label_names[0], ascending=False)

        s = dataset.protected_attribute_names[0]
        self._item_groups = {ai: it for ai, it in zip(
            s_vals, [items[items[s] == ai] for ai in s_vals])}
        counts_a = {a: 0 for a in s_vals}
        rankedItems = []

        if rerank_type != 'Constrained':
            for k in range(1, rec_size+1):
                below_min, below_max = [], []
                candidates = [
                    candidates_ai.iloc[counts_a[ai]] for ai, candidates_ai in self._item_groups.items()
                    ]
                for ai in s_vals:
                    # best unranked items for each sensitive attribute
                    if counts_a[ai] < np.floor(k*target_prop[ai]):
                        below_min.append((ai))
                    elif counts_a[ai] < np.ceil(k*target_prop[ai]):
                        below_max.append((ai))
                if len(below_min) != 0:
                    candidates_bmin = [c for c in candidates if c[s] in below_min]
                    next_item = max(candidates_bmin, key = lambda x: x['score'])
                else:
                    # if Greedy, add the highest scoring candidate among the groups
                    if rerank_type == 'Greedy':
                        candidates_bmax = [c for c in candidates if c[s] in below_max]
                        next_item = max(candidates_bmax, key = lambda x: x['score'])
                    # if Conservative, add the candidate from the group least represented so far
                    elif rerank_type == 'Conservative':
                        next_attr = min(below_max, key=lambda ai:
                                        np.ceil(k*target_prop[ai])/target_prop[ai])
                        next_item = self._item_groups[next_attr].iloc[counts_a[next_attr]]
                    # if Relaxed, relax the conservative requirements
                    elif rerank_type == 'Relaxed':
                        next_attr_set = min(below_max, key=lambda ai:
                                            np.ceil(np.ceil(k*target_prop[ai])/target_prop[ai]))
                        if not isinstance(next_attr_set, list):
                            next_attr_set = [next_attr_set]
                        candidates_rel = [c for c in candidates if c[s] in next_attr_set]
                        # best item among best items for each attribute in next_attr_set
                        next_item = max(candidates_rel, key=lambda x: x['score'])

                rankedItems.append(next_item)
                counts_a[next_item[s]] += 1

        elif rerank_type == 'Constrained':
            rankedItems, maxIndices = [], []
            counts_a, min_counts = {a: 0 for a in s_vals}, {a: 0 for a in s_vals}
            lastEmpty, k = 0, 0
            while lastEmpty < rec_size:
                k+=1
                # determine the minimum feasible counts of each group at current rec. list size
                min_counts_at_k = {ai: int(pai*k) for ai, pai in target_prop.items()}
                # get sensitive attr. values for which the current minimum count has increased
                # since last one
                changed_mins = []
                for ai in s_vals:
                    if min_counts_at_k[ai] > min_counts[ai]:
                        changed_mins.append(ai)
                    
                if len(changed_mins) > 0:
                    # get the list of candidates to insert and sort them by their score
                    changed_items = []
                    for ai in changed_mins:
                        changed_items.append(self._item_groups[ai].iloc[counts_a[ai]])
                    changed_items.sort(key=lambda x: x['score'])

                    # add the items, starting with the best score
                    for newitem in changed_items:
                        maxIndices.append(k)
                        rankedItems.append(newitem)
                        start = lastEmpty
                        while start > 0 and maxIndices[start-1] >= start and rankedItems[start-1]['score'] < rankedItems[start]['score']:
                            maxIndices[start-1], maxIndices[start] = maxIndices[start], maxIndices[start-1]
                            rankedItems[start-1], rankedItems[start] = rankedItems[start], rankedItems[start-1]
                            start -= 1
                        lastEmpty+=1
                        counts_a[newitem[s]] += 1
                    min_counts = min_counts_at_k

        res_df = pd.DataFrame(rankedItems, columns=dataset.feature_names + ['score'])
        res = RegressionDataset(res_df,
                                dep_var_name=dataset.label_names[0],
                                protected_attribute_names=dataset.protected_attribute_names,
                                privileged_classes=dataset.privileged_protected_attributes)
        if not normalize_scores:
            res.labels = np.transpose([res_df['score']])
        return res