from aif360.metrics.mdss.ScoringFunctions.ScoringFunction import ScoringFunction
from aif360.metrics.mdss.generator import get_entire_subset, get_random_subset

import pandas as pd
import numpy as np


class MDSS(object):

    def __init__(self, scoring_function: ScoringFunction):
        self.scoring_function = scoring_function

    def get_aggregates(self, coordinates: pd.DataFrame, outcomes: pd.Series, probs: pd.Series,
                       current_subset: dict, column_name: str, penalty: float):
        """
        Conditioned on the current subsets of values for all other attributes,
        compute the summed outcome (observed_sum = \sum_i y_i) and all probabilities p_i
        for each value of the current attribute.
        Also use additive linear-time subset scanning to compute the set of distinct thresholds
        for which different subsets of attribute values have positive scores. Note that the number
        of such thresholds will be linear rather than exponential in the arity of the attribute.

        :param coordinates: data frame containing having as columns the covariates/features
        :param probs: data series containing the probabilities/expected outcomes
        :param outcomes: data series containing the outcomes/observed outcomes
        :param current_subset: current subset to compute aggregates
        :param column_name: attribute name to scan over
        :param penalty: penalty coefficient
        :return: dictionary of aggregates, sorted thresholds (roots), observed sum of the subset, array of observed
        probabilities
        """

        # compute the subset of records matching the current subgroup along all other dimensions
        # temp_df includes the covariates x_i, outcome y_i, and predicted probability p_i for each matching record
        if current_subset:
            to_choose = coordinates[current_subset.keys()].isin(current_subset).all(axis=1)
            temp_df = pd.concat([coordinates.loc[to_choose], outcomes[to_choose], probs[to_choose]], axis=1)
        else:
            temp_df = pd.concat([coordinates, outcomes, probs], axis=1)

        # these wil be used to keep track of the aggregate values and the distinct thresholds to be considered
        aggregates = {}
        thresholds = set()

        scoring_function = self.scoring_function

        # consider each distinct value of the given attribute (column_name)
        for name, group in temp_df.groupby(column_name):
            # compute the sum of outcomes \sum_i y_i
            observed_sum = group.iloc[:, -2].sum()

            # all probabilities p_i
            probs = group.iloc[:, -1].values

            # compute q_min and q_max for the attribute value
            exist, q_mle, q_min, q_max = scoring_function.compute_qs(observed_sum, probs, penalty)

            # Add to aggregates, and add q_min and q_max to thresholds.
            # Note that thresholds is a set so duplicates will be removed automatically.
            if exist:
                aggregates[name] = {
                    'q_mle': q_mle,
                    'q_min': q_min,
                    'q_max': q_max,
                    'observed_sum': observed_sum,
                    'probs': probs
                }
                thresholds.update([q_min, q_max])

        # We also keep track of the summed outcomes \sum_i y_i and the probabilities p_i for the case where _
        # all_ values of that attribute are considered (regardless of whether they contribute positively to score).
        # This is necessary because of the way we compute the penalty term: including all attribute values, equivalent
        # to ignoring the attribute, has the lowest penalty (of 0) and thus we need to score that subset as well.
        all_observed_sum = temp_df.iloc[:, -2].sum()
        all_probs = temp_df.iloc[:, -1].values

        return [aggregates, sorted(thresholds), all_observed_sum, all_probs]

    def choose_aggregates(self, aggregates: dict, thresholds: list, penalty: float, all_observed_sum: float,
                          all_probs: list):
        """
        Having previously computed the aggregates and the distinct q thresholds
        to consider in the get_aggregates function,we are now ready to choose the best
        subset of attribute values for the given attribute.
        For each range defined by these thresholds, we will choose all of the positive contributions,
        compute the MLE value of q, and the corresponding score.
        We then pick the best q and score over all of the ranges considered.

        :param aggregates: dictionary of aggregates. For each feature value, it has q_mle, q_min, q_max, observed_sum,
        and the probabilities
        :param thresholds: sorted thresholds (roots)
        :param penalty: penalty coefficient
        :param all_observed_sum: sum of observed binary outcomes for all i
        :param all_probs: data series containing all the probabilities/expected outcomes
        :return:
        """
        # initialize
        best_score = 0
        best_names = []

        scoring_function = self.scoring_function

        # for each threshold
        for i in range(len(thresholds) - 1):
            threshold = (thresholds[i] + thresholds[i + 1]) / 2
            observed_sum = 0.0
            probs = []
            names = []

            # keep only the aggregates which have a positive contribution to the score in that q range
            # we must keep track of the sum of outcome values as well as all predicted probabilities
            for key, value in aggregates.items():
                if (value['q_min'] < threshold) & (value['q_max'] > threshold):
                    names.append(key)
                    observed_sum += value['observed_sum']
                    probs = probs + value['probs'].tolist()

            if len(probs) == 0:
                continue

            # compute the MLE value of q, making sure to only consider the desired direction (positive or negative)
            probs = np.asarray(probs)
            current_q_mle = scoring_function.qmle(observed_sum, probs)

            # Compute the score for the given subset at the MLE value of q.
            # Notice that each included value gets a penalty, so the total penalty
            # is multiplied by the number of included values.
            current_interval_score = scoring_function.score(observed_sum, probs, penalty * len(names), current_q_mle)

            # keep track of the best score, best q, and best subset of attribute values found so far
            if current_interval_score > best_score:
                best_score = current_interval_score
                best_names = names

        # Now we also have to consider the case of including all attribute values,
        # including those that never make positive contributions to the score.
        # Note that the penalty term is 0 in this case.  (We are neglecting penalties
        # from all other attributes, just considering the current attribute.)

        # compute the MLE value of q, making sure to only consider the desired direction (positive or negative)
        current_q_mle = scoring_function.qmle(all_observed_sum, all_probs)

        # Compute the score for the given subset at the MLE value of q.
        # Again, the penalty (for that attribute) is 0 when all attribute values are included.
        
        current_score = scoring_function.score(all_observed_sum, all_probs, 0, current_q_mle)

        # Keep track of the best score, best q, and best subset of attribute values found.
        # Note that if the best subset contains all values of the given attribute,
        # we return an empty list for best_names.
        if current_score > best_score:
            best_score = current_score
            best_names = []

        return [best_names, best_score]

    def score_current_subset(self, coordinates: pd.DataFrame, probs: pd.Series, outcomes: pd.Series,
                             current_subset: dict, penalty: float):
        """
        Just scores the subset without performing ALTSS.
        We still need to determine the MLE value of q.

        :param coordinates: data frame containing having as columns the covariates/features
        :param probs: data series containing the probabilities/expected outcomes
        :param outcomes: data series containing the outcomes/observed outcomes
        :param current_subset: current subset to be scored
        :param penalty: penalty coefficient
        :return: penalized score of subset
        """

        # compute the subset of records matching the current subgroup along all dimensions
        # temp_df includes the covariates x_i, outcome y_i, and predicted probability p_i for each matching record
        if current_subset:
            to_choose = coordinates[current_subset.keys()].isin(current_subset).all(axis=1)
            temp_df = pd.concat([coordinates.loc[to_choose], outcomes[to_choose], probs[to_choose]], axis=1)
        else:
            temp_df = pd.concat([coordinates, outcomes, probs], axis=1)

        scoring_function = self.scoring_function

        # we must keep track of the sum of outcome values as well as all predicted probabilities
        observed_sum = temp_df.iloc[:, -2].sum()
        probs = temp_df.iloc[:, -1].values

        # compute the MLE value of q, making sure to only consider the desired direction (positive or negative)
        current_q_mle = scoring_function.qmle(observed_sum, probs)

        # total_penalty = penalty * sum of list lengths in current_subset
        total_penalty = 0
        for key, values in current_subset.items():
            total_penalty += len(values)

        total_penalty *= penalty

        # Compute and return the penalized score    
        penalized_score = scoring_function.score(observed_sum, probs, total_penalty, current_q_mle)
        return penalized_score

    def scan(self, coordinates: pd.DataFrame, probs: pd.Series, outcomes: pd.Series, penalty: float,
                    num_iters: int, verbose: bool = False, seed: int = 0):
        """
        :param coordinates: data frame containing having as columns the covariates/features
        :param probs: data series containing the probabilities/expected outcomes
        :param outcomes: data series containing the outcomes/observed outcomes
        :param penalty: penalty coefficient
        :param num_iters: number of iteration
        :param verbose: logging flag
        :param seed: numpy seed. Default equals 0
        :return: [best subset, best score]
        """
        np.random.seed(seed)

        # initialize
        best_subset = {}
        best_score = -1e10
        best_scores = []
        for i in range(num_iters):
            # flags indicates that the method has optimized over subsets for a given attribute.
            # The iteration ends when it cannot further increase score by optimizing over
            # subsets of any attribute, i.e., when all flags are 1.
            flags = np.empty(len(coordinates.columns))
            flags.fill(0)

            # Starting subset. Note that we start with all values for the first iteration
            # and random values for succeeding iterations.
            current_subset = get_entire_subset() if (i == 0) \
                else get_random_subset(coordinates, np.random.rand(1).item(), 10)

            # score the entire population
            current_score = self.score_current_subset(
                coordinates=coordinates,
                probs=probs,
                outcomes=outcomes,
                penalty=penalty,
                current_subset=current_subset
            )

            while flags.sum() < len(coordinates.columns):

                # choose random attribute that we haven't scanned yet
                attribute_number_to_scan = np.random.choice(len(coordinates.columns))
                while flags[attribute_number_to_scan]:
                    attribute_number_to_scan = np.random.choice(len(coordinates.columns))
                attribute_to_scan = coordinates.columns.values[attribute_number_to_scan]

                # clear current subset of attribute values for that subset
                if attribute_to_scan in current_subset:
                    del current_subset[attribute_to_scan]

                # call get_aggregates and choose_aggregates to find best subset of attribute values
                aggregates, thresholds, all_observed_sum, all_probs = self.get_aggregates(
                    coordinates=coordinates,
                    outcomes=outcomes,
                    probs=probs,
                    current_subset=current_subset,
                    column_name=attribute_to_scan,
                    penalty=penalty
                )

                temp_names, temp_score = self.choose_aggregates(
                    aggregates=aggregates,
                    thresholds=thresholds,
                    penalty=penalty,
                    all_observed_sum=all_observed_sum,
                    all_probs=all_probs
                )

                temp_subset = current_subset.copy()
                # if temp_names is not empty (or null)
                if temp_names:
                    temp_subset[attribute_to_scan] = temp_names

                # Note that this call to score_current_subset ensures that
                # we are penalizing complexity for all attribute values.
                # The value of temp_score computed by choose_aggregates
                # above includes only the penalty for the current attribute.
                temp_score = self.score_current_subset(
                    coordinates=coordinates,
                    probs=probs,
                    outcomes=outcomes,
                    penalty=penalty,
                    current_subset=temp_subset
                )

                # reset flags to 0 if we have improved score
                if temp_score > current_score + 1E-6:
                    flags.fill(0)

                # TODO: confirm with Skyler: sanity check to make sure score has not decreased
                assert temp_score >= current_score - 1E-6, \
                    "WARNING SCORE HAS DECREASED from %.3f to %.3f" % (current_score, temp_score)

                flags[attribute_number_to_scan] = 1
                current_subset = temp_subset
                current_score = temp_score

            # print out results for current iteration
            if verbose:
                print("Subset found on iteration", i + 1, "of", num_iters, "with score", current_score, ":")
                print(current_subset)

            # update best_score and best_subset if necessary
            if current_score > best_score:
                best_subset = current_subset.copy()
                best_score = current_score

                if verbose:
                    print("Best score is now", best_score)

            elif verbose:
                print("Current score of", current_score, "does not beat best score of", best_score)
            best_scores.append(best_score)
        return best_subset, best_score
