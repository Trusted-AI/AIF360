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

        super(DeterministicReranking, self).__init__(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)
        
        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups
        self._n_groups = len(unprivileged_groups) + len(privileged_groups)
        self.s = list(unprivileged_groups[0].keys())[0]
        self.s_vals = set(self.unprivileged_groups[0].values()).union(set(self.privileged_groups[0].values()))

    def fit(self, dataset: RegressionDataset):

        if len(self.unprivileged_groups) != 1 or len(self.privileged_groups) != 1:
            raise ValueError("Only one unprivileged group or privileged group supported.")
        if list(self.unprivileged_groups[0].keys())[0] != list(self.privileged_groups[0].keys())[0]:
            raise ValueError("Different sensitive attributes (not values) specified for unprivileged and privileged groups.")  

        items = dataset.convert_to_dataframe()[0]
        items = items.sort_values(axis=0, by=dataset.label_names[0], ascending=False)

        if self.s not in items.columns:
            raise ValueError(f"The dataset must contain the protected attribute: '{self.s}'.")
        
        # if we have just 1 protected attribute
        if not isinstance(self.s, list) and False:
            self._item_groups = [items[items[self.s] == ai] for ai in self.s_vals]
        else:
            self._item_groups = []
            for group in self.unprivileged_groups + self.privileged_groups:
                q = ' & '.join(
                    [f'{s_i}=="{v_i}"' if isinstance(v_i, str) else f'{s_i}=={v_i}' 
                     for s_i, v_i in group.items()]
                     )
                self._item_groups.append(items.query(q))

        # self._item_groups = {ai: it for ai, it in zip(
        #     self.s_vals, [items[items[self.s] == ai] for ai in self.s_vals])}
        
        return self

    def predict(self,
                    dataset: RegressionDataset,
                    rec_size: int,
                    target_prop: dict,
                    rerank_type: str='Constrained',
                    renormalize_scores: bool=False
                    ) -> RegressionDataset:
        """Construct a ranking of candidates in the dataset according to specified proportions of groups.

        Args:
            dataset (RegressionDataset): Dataset to rerank.
            rec_size (int): Number of candidates in the output.
            target_prop (dict): Desired proportion of groups in the output. Format is \
                {protected_attribute_value: proportion}.
            rerank_type: Greedy, Conservative, Relaxed, or Constrained. Determines the type of algorithm \
                as described in the original paper.
            renormalize_scores: renormalize label (score) values in the resulting ranking. If True, uses the default \
                behavior of RegressionDataset.

        Returns:
            RegressionDataset: The reranked dataset.
        """
        
        
        if rec_size <= 0:
            raise ValueError(f"Output size should be greater than 0, got {rec_size}.")
        # if np.any(set(target_prop.keys()) != set(self.s_vals)):
        #     raise ValueError("""Proportion specifications do not match. \
        #         `target_prop` should have sensitive attribute values as keys.""")
        if len(dataset.label_names) != 1:
            raise ValueError(f"Dataset must have exactly one label, got {len(dataset.label_names)}.")
        if rerank_type not in ['Greedy', 'Conservative', 'Relaxed', 'Constrained']:
            raise ValueError(f'`rerank_type` must be one of `Greedy`, `Conservative`, `Relaxed`, `Constrained`; got {rerank_type}')
        
        # group_counts = {a: 0 for a in self.s_vals}
        group_counts = [0] * self._n_groups
        rankedItems = []
        score_label = dataset.label_names[0]

        if rerank_type != 'Constrained':
            for k in range(1, rec_size+1):
                below_min, below_max = [], []
                # get the best-scoring candidate item from each group
                candidates = [
                    candidates_gi.iloc[group_counts[g_i]] for g_i, candidates_gi in enumerate(self._item_groups)
                    ]
                for group_idx in range(self._n_groups):
                    # best unranked items for each group
                    if group_counts[group_idx] < np.floor(k*target_prop[group_idx]):
                        below_min.append(group_idx)
                    elif group_counts[group_idx] < np.ceil(k*target_prop[group_idx]):
                        below_max.append(group_idx)
                # if some groups are currently underrepresented
                if len(below_min) != 0:
                    # choose the best next item among currently underrepresented groups
                    candidates_bmin = [candidates[group_idx] for group_idx in below_min]
                    next_group, next_item = max(enumerate(candidates_bmin), key = lambda x: x[1][score_label])
                # if minimal representation requirements are satisfied
                else:
                    # if Greedy, add the highest scoring candidate among the groups
                    if rerank_type == 'Greedy':
                        candidates_bmax = [candidates[group_idx] for group_idx in below_max]
                        next_group, next_item = max(enumerate(candidates_bmax), key = lambda x: x[1][score_label])
                    # if Conservative, add the candidate from the group least represented so far
                    elif rerank_type == 'Conservative':
                        # group_rep = [np.ceil(k*target_prop[group])/target_prop[group] for group in below_max]
                        # sort by how close the groups are to violating the condition, in case of tie sort by best element score
                        next_group = min(below_max, key=lambda group_idx:
                                        (np.ceil(k*target_prop[group_idx])/target_prop[group_idx],
                                         -candidates[group_idx][score_label]))
                        next_item = candidates[next_group]
                    # if Relaxed, relax the conservative requirements
                    elif rerank_type == 'Relaxed':
                        next_group = min(below_max, key=lambda group_idx:
                                            (np.ceil(np.ceil(k*target_prop[group_idx])/target_prop[group_idx]),
                                             -candidates[group_idx][score_label])
                                             )
                        next_item = candidates[next_group]

                rankedItems.append(next_item)
                group_counts[next_group] += 1

        elif rerank_type == 'Constrained':
            rankedItems, maxIndices = [], []
            group_counts, min_counts = [0] * self._n_groups, [0] * self._n_groups

            # group_counts, min_counts = {a: 0 for a in self.s_vals}, {a: 0 for a in self.s_vals}
            lastEmpty, k = 0, 0
            while lastEmpty < rec_size:
                k+=1
                # determine the minimum feasible counts of each group at current rec. list size
                min_counts_at_k = [int(p_gi*k) for p_gi in target_prop]
                # min_counts_at_k = {ai: int(pai*k) for ai, pai in target_prop.items()}
                # get sensitive attr. values for which the current minimum count has increased
                # since last one
                changed_mins = []
                for group_idx in range(self._n_groups):
                    if min_counts_at_k[group_idx] > min_counts[group_idx]:
                        changed_mins.append(group_idx)
                    
                if len(changed_mins) > 0:
                    # get the list of candidates to insert and sort them by their score
                    changed_items = []
                    # save the candidate AND the index of the group it belongs to
                    for group_idx in changed_mins:
                        changed_items.append((group_idx, self._item_groups[group_idx].iloc[group_counts[group_idx]]))
                    changed_items.sort(key=lambda x: x[1][score_label])

                    # add the candidate items, starting with the best score
                    for newitem in changed_items:
                        maxIndices.append(k)
                        rankedItems.append(newitem[1])
                        start = lastEmpty
                        while start > 0 and maxIndices[start-1] >= start and rankedItems[start-1][score_label] < rankedItems[start][score_label]:
                            maxIndices[start-1], maxIndices[start] = maxIndices[start], maxIndices[start-1]
                            rankedItems[start-1], rankedItems[start] = rankedItems[start], rankedItems[start-1]
                            start -= 1
                        lastEmpty+=1
                        group_counts[newitem[0]] += 1
                    min_counts = min_counts_at_k

        res_df = pd.DataFrame(rankedItems, columns=dataset.feature_names + [score_label])
        res = RegressionDataset(res_df,
                                dep_var_name=dataset.label_names[0],
                                protected_attribute_names=dataset.protected_attribute_names,
                                privileged_classes=dataset.privileged_protected_attributes)
        if not renormalize_scores:
            res.labels = np.transpose([res_df[score_label]])
        return res
    

    def fit_predict(self,
                    dataset: RegressionDataset,
                    rec_size: int,
                    target_prop: dict,
                    rerank_type: str='Constrained',
                    renormalize_scores: bool=False
                    ) -> RegressionDataset:
        self.fit(dataset=dataset)
        return self.predict(dataset=dataset, 
                            rec_size=rec_size, 
                            target_prop=target_prop, 
                            rerank_type=rerank_type, 
                            renormalize_scores=renormalize_scores)