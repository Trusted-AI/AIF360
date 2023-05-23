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
            unprivileged_groups (list(dict)): Representation for unprivileged
                group.
            privileged_groups (list(dict)): Representation for privileged
                group.
        """
        if len(unprivileged_groups) != 1 or len(privileged_groups) != 1:
            raise ValueError("Only one unprivileged_group or privileged_group supported.")
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
                    target_prop: dict | list,
                    rerank_type: str='Constrained',
                    normalize_scores: bool=False
                    ):
        """Rerank the dataset according to previously specified proportions of groups.

        Args:
            dataset (RegressionDataset): Dataset to rerank.
            rerank_type: Greedy, Conservative, Relaxed, or Constrained. Determines the type of algorithm \
                as described in the original paper.
            normalize_scores: normalize label (score) values in the resulting ranking. If True, uses the default \
                behavior of RegressionDataset.

        Returns:
            RegressionDataset: The reranked dataset.
        """
        
        s_vals = set(self.unprivileged_groups[0].values()).union(set(self.privileged_groups[0].values()))

        if rec_size <= 0:
            raise ValueError(f"Output size should be greater or equal to 0, got {rec_size}.")
        if isinstance(target_prop, dict):
            if np.any(set(target_prop.keys()) != set(s_vals)):
                raise ValueError("""Proportion specifications do not match. \
                    target_prop should have sensitive attribute values as keys!""")
        elif isinstance(target_prop, list):
            if len(target_prop) != 2:
                raise ValueError("""Proportion specifications do not match. \
                    target_prop length should be equal to the number of sensitive attribute values!""")
            
        target_prop_ = target_prop if isinstance(target_prop, dict) else {
            ai: pi for ai, pi in zip(s_vals, target_prop)}

        if len(dataset.label_names) != 1:
            raise ValueError("There must be exactly one label.")
        
        items = dataset.convert_to_dataframe()[0]
        items = items.sort_values(axis=0, by=dataset.label_names[0], ascending=False)

        s = dataset.protected_attribute_names[0]
        self._item_groups = {ai: it for ai, it in zip(
            s_vals, [items[items[s] == ai] for ai in s_vals])}
        counts_a = {a: 0 for a in s_vals}
       
        rankedItems = []
        if rerank_type != 'Constrained':
            for k in range(1, rec_size+1):
                below_min = []
                below_max = []
                candidates = [
                    candidates_ai.iloc[counts_a[ai]] for ai, candidates_ai in self._item_groups.items()
                    ]
                for ai in s_vals:
                    # best unranked items for each sensitive attribute
                    if counts_a[ai] < np.floor(k*target_prop_[ai]):
                        below_min.append((ai))
                    elif counts_a[ai] < np.ceil(k*target_prop_[ai]):
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
                                        np.ceil(k*target_prop_[ai])/target_prop_[ai])
                        next_item = self._item_groups[next_attr].iloc[counts_a[next_attr]]
                    # if Relaxed, relax the conservative requirements
                    elif rerank_type == 'Relaxed':
                        next_attr_set = min(below_max, key=lambda ai:
                                            np.ceil(np.ceil(k*target_prop_[ai])/target_prop_[ai]))
                        if not isinstance(next_attr_set, list):
                            next_attr_set = [next_attr_set]
                        candidates_rel = [c for c in candidates if c[s] in next_attr_set]
                        # best item among best items for each attribute in next_attr_set
                        next_item = max(candidates_rel, key=lambda x: x['score'])

                rankedItems.append(next_item)
                counts_a[next_item[s]] += 1

        elif rerank_type == 'Constrained':
            rankedItems = []
            maxIndices = []
            counts_a = {a: 0 for a in s_vals}
            min_counts = {a: 0 for a in s_vals}
            lastEmpty = 0
            k=0
            while lastEmpty < rec_size:
                k+=1
                # determine the minimum feasible counts of each group at current rec. list size
                min_counts_at_k = {ai: int(pai*k) for ai, pai in target_prop_.items()}
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


    # def fit(self, dataset_pred, dataset_true=None):
    #     """Compute parameters for reranking utilizing the given ranking.

    #     Args:
    #         dataset_pred (RegressionDataset): Dataset containing the predicted ranking.

    #     Returns:
    #         DeterministicReranking: Returns self.
    #     """
    #     if len(dataset_pred.label_names) != 1:
    #         raise ValueError("There must be exactly one label.") 
    #     items = pd.DataFrame(dataset_pred.features, columns=dataset_pred.feature_names)
    #     items = items.sort_values(axis=0, by=dataset_pred.label_names[0], ascending=False)

    #     priv_group, unpriv_group = self.privileged_groups[0], self.unprivileged_groups[0]
    #     s_vals = list(priv_group.values()).append(list(unpriv_group.values()))
    #     s = list(priv_group.keys())[0]
    #     self._item_groups = {ai: it for ai, it in zip(
    #         s_vals, [items[items[s] == ai] for ai in s_vals])}
    
    # def predict(self, dataset=None, rerank_type='Constrained'):
    #     """.

    #     Args:
    #         dataset (RegressionDataset): Dataset to rerank.
    #         rerank_type: Greedy, Conservative, Relaxed, or Constrained. Determines the type of algorithm
    #             as described in the original paper.

    #     Returns:
    #         RegressionDataset: Returns reranked dataset.
    #     """
    #     rankedItems = []
        
    #     priv_group, unpriv_group = self.privileged_groups[0], self.unprivileged_groups[0]
    #     s_vals = list(priv_group.values()).append(list(unpriv_group.values()))
    #     s = list(priv_group.keys())[0]

    #     counts_a = {a: 0 for a in s_vals}
       
    #     if rerank_type == 'Greedy':
    #         for k in range(1, rec_size):
    #             below_min = []
    #             below_max = []
    #             candidates = [
    #                 candidates_ai.iloc[counts_a[ai]] for ai, candidates_ai in self._item_groups.items()
    #                 ]
    #             for ai in s_vals:
    #                 # best unranked items for each sensitive attribute
    #                 if counts_a[ai] < np.floor(k*target_prop_[ai]):
    #                     below_min.append((ai))
    #                 elif counts_a[ai] < np.ceil(k*target_prop_[ai]):
    #                     below_max.append((ai))
    #             if len(below_min) != 0:
    #                 candidates_bmin = [c for c in candidates if c[s] in below_min]
    #                 next_item = max(candidates_bmin, key = lambda x: x['score'])
    #             else:
    #                 candidates_bmax = [c for c in candidates if c[s] in below_max]
    #                 next_item = max(candidates_bmax, key = lambda x: x['score'])
    #             rankedItems.append(next_item)
    #             counts_a[next_item[s]] += 1
    #         return RegressionDataset(pd.DataFrame(rankedItems, columns=))






def detgreedy(items: pd.DataFrame, s: str, 
                 props: dict, kmax: int=10):
    rankedItems = []
    counts_a = {a: 0 for a in props.keys()}
    #counts_a = np.zeros(shape=len(probs.keys()))
    
    items = items.sort_values(axis=0, by='score',ascending=False)
    item_groups = {ai: it for ai, it in zip(props.keys(), [items[items[s] == ai] for ai in props.keys()])}

    for k in range(1,kmax):
        below_min = []
        below_max = []
        candidates = [candidates_ai.iloc[counts_a[ai]] for ai, candidates_ai in item_groups.items()]
        for ai in props.keys():
            # best unranked items for each sensitive attribute
            if counts_a[ai] < np.floor(k*props[ai]):
                below_min.append((ai))
            elif counts_a[ai] < np.ceil(k*props[ai]):
                below_max.append((ai))
        if len(below_min) != 0:
            candidates_bmin = [c for c in candidates if c[s] in below_min]
            next_item = max(candidates_bmin, key = lambda x: x['score'])
        else:
            candidates_bmax = [c for c in candidates if c[s] in below_max]
            next_item = max(candidates_bmax, key = lambda x: x['score'])
        rankedItems.append(next_item)
        counts_a[next_item[s]] += 1
    return pd.DataFrame(rankedItems)

def detcons(items: pd.DataFrame, s: str, 
                 props: dict, kmax: int=10, relaxed=False):
    rankedItems = []
    counts_a = {a: 0 for a in props.keys()}
    #counts_a = np.zeros(shape=len(probs.keys()))
    
    items = items.sort_values(axis=0, by='score',ascending=False)
    item_groups = {ai: it for ai, it in zip(props.keys(), [items[items[s] == ai] for ai in props.keys()])}

    for k in range(1,kmax):
        below_min = []
        below_max = []
        candidates = [candidates_ai.iloc[counts_a[ai]] for ai, candidates_ai in item_groups.items()]
        for ai in props.keys():
            # best unranked items for each sensitive attribute
            if counts_a[ai] < np.floor(k*props[ai]):
                below_min.append((ai))
            elif counts_a[ai] < np.ceil(k*props[ai]):
                below_max.append((ai))
        if len(below_min) != 0:
            candidates_bmin = [c for c in candidates if c[s] in below_min]
            next_item = max(candidates_bmin, key = lambda x: x['score'])
        else:
            # sort by scores if tie in lambda?
            if relaxed:
                next_attr_set = min(below_max, key=lambda ai: np.ceil(np.ceil(k*props[ai])/props[ai]))
                if not isinstance(next_attr_set, list):
                    next_attr_set = [next_attr_set]
                candidates_rel = [c for c in candidates if c[s] in next_attr_set]
                # best item among best items for each attribute in next_attr_set
                next_item = max(candidates_rel, key=lambda x: x['score'])
            else:
                next_attr = min(below_max, key=lambda ai: np.ceil(k*props[ai])/props[ai])
                next_item = item_groups[next_attr].iloc[counts_a[next_attr]]
        rankedItems.append(next_item)
        counts_a[next_item[s]] += 1
    return pd.DataFrame(rankedItems)

def detconstsort(items: pd.DataFrame | dict, s: str, 
                 props: dict, kmax: int=10,):
    rankedItems = []
    maxIndices = []
    sens_vals = list(props.keys())
    counts_a = {a: 0 for a in sens_vals}
    min_counts = {a: 0 for a in sens_vals}
    lastEmpty = 0
    k=0
    # check if there's enough items of each class to satisfy constraint
    for ai in sens_vals:
        target_count = kmax*props[ai]
        if target_count > items[items[s] == ai].shape[0]:
            raise ValueError(
                f'Not enough items of group -- {ai} -- to satisfy target constraint!'
            )
    # group the items by their sensitive attribute values
    if not isinstance(items, dict):
        items = items.sort_values(axis=0, by='score',ascending=False)
        item_groups = {ai: it for ai, it in zip(sens_vals, [items[items[s] == ai] for ai in sens_vals])}
    else:
        # check if dict is constructed correctly maybe?
        raise(NotImplementedError)
    while lastEmpty < kmax:
        k+=1
        # determine the minimum feasible counts of each group at current rec. list size
        min_counts_at_k = {ai: int(pai*k) for ai, pai in props.items()}
        # get sensitive attr. values for which the current minimum count has increased
        # since last one
        changed_mins = []
        for ai in sens_vals:
            if min_counts_at_k[ai] > min_counts[ai]:
                changed_mins.append(ai)
            
        if len(changed_mins) > 0:
            # get the list of candidates to insert and sort them by their score
            changed_items = []
            for ai in changed_mins:
                changed_items.append(item_groups[ai].iloc[counts_a[ai]])
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
    return pd.DataFrame(rankedItems)