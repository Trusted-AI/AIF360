from aif360.explainers import Explainer
from aif360.metrics import Metric


class MetricTextExplainer(Explainer):
    """Class for explaining metric values with text.

    These briefly explain what a metric is and/or how it is calculated unless it
    is obvious (e.g. accuracy) and print the value.

    This class contains text explanations for all metric values regardless of
    which subclass they appear in. This will raise an error if the metric does
    not apply (e.g. calling `true_positive_rate` if
    `type(metric) == DatasetMetric`).
    """

    def __init__(self, metric):
        """Initialize a `MetricExplainer` object.

        Args:
            metric (Metric): The metric to be explained.
        """
        if isinstance(metric, Metric):
            self.metric = metric
        else:
            raise TypeError("metric must be a Metric.")

    def accuracy(self, privileged=None):
        if privileged is None:
            return "Classification accuracy (ACC): {}".format(
                self.metric.accuracy(privileged=privileged))
        return "Classification accuracy on {} instances: {}".format(
            'privileged' if privileged else 'unprivileged',
            self.metric.accuracy(privileged=privileged))

    def average_abs_odds_difference(self):
        return ("Average absolute odds difference (average of abs(TPR "
                "difference) and abs(FPR difference)): {}".format(
                    self.metric.average_abs_odds_difference()))

    def average_odds_difference(self):
        return ("Average odds difference (average of TPR difference and FPR "
                "difference, 0 = equality of odds): {}".format(
                    self.metric.average_odds_difference()))

    def between_all_groups_coefficient_of_variation(self):
        return "Between-group coefficient of variation: {}".format(
            self.metric.between_all_groups_coefficient_of_variation())

    def between_all_groups_generalized_entropy_index(self, alpha=2):
        return "Between-group generalized entropy index: {}".format(
            self.metric.between_all_groups_generalized_entropy_index(alpha=alpha))

    def between_all_groups_theil_index(self):
        return "Between-group Theil index: {}".format(
            self.metric.between_all_groups_theil_index())

    def between_group_coefficient_of_variation(self):
        return "Between-group coefficient of variation: {}".format(
            self.metric.between_group_coefficient_of_variation())

    def between_group_generalized_entropy_index(self, alpha=2):
        return "Between-group generalized entropy index: {}".format(
            self.metric.between_group_generalized_entropy_index(alpha=alpha))

    def between_group_theil_index(self):
        return "Between-group Theil index: {}".format(
            self.metric.between_group_theil_index())

    def coefficient_of_variation(self):
        return "Coefficient of variation: {}".format(
            self.metric.coefficient_of_variation())

    def consistency(self, n_neighbors=5):
        return "Consistency (Zemel, et al. 2013): {}".format(
            self.metric.consistency(n_neighbors=n_neighbors))

    def disparate_impact(self):
        return ("Disparate impact (probability of favorable outcome for "
                "unprivileged instances / probability of favorable outcome for "
                "privileged instances): {}".format(
                    self.metric.disparate_impact()))

    def error_rate(self, privileged=None):
        if privileged is None:
            return "Error rate (ERR = 1 - ACC): {}".format(
                self.metric.error_rate(privileged=privileged))
        return "Error rate on {} instances: {}".format(
            'privileged' if privileged else 'unprivileged',
            self.metric.error_rate(privileged))

    def error_rate_difference(self):
        return ("Error rate difference (error rate on unprivileged instances - "
                "error rate on privileged instances): {}".format(
                    self.metric.error_rate_difference()))

    def error_rate_ratio(self):
        return ("Error rate ratio (error rate on unprivileged instances / "
                "error rate on privileged instances): {}".format(
                    self.metric.error_rate_ratio()))

    def false_discovery_rate(self, privileged=None):
        if privileged is None:
            return "False discovery rate (FDR = FP / (FP + TP)): {}".format(
                self.metric.false_discovery_rate(privileged=privileged))
        return "False discovery rate on {} instances: {}".format(
            'privileged' if privileged else 'unprivileged',
            self.metric.false_discovery_rate(privileged=privileged))

    def false_discovery_rate_difference(self):
        return ("False discovery rate difference (false discovery rate on "
                "unprivileged instances - false discovery rate on privileged "
                "instances): {}".format(
                    self.metric.false_discovery_rate_difference()))

    def false_discovery_rate_ratio(self):
        return ("False discovery rate ratio (false discovery rate on "
                "unprivileged instances - false discovery rate on privileged "
                "instances): {}".format(
                    self.metric.false_discovery_rate_ratio()))

    def false_negative_rate(self, privileged=None):
        if privileged is None:
            return "False negative rate (FNR = FN / (TP + FN)): {}".format(
                self.metric.false_negative_rate(privileged=privileged))
        return "False negative rate on {} instances: {}".format(
            'privileged' if privileged else 'unprivileged',
            self.metric.false_negative_rate(privileged=privileged))

    def false_negative_rate_difference(self):
        return ("False negative rate difference (false negative rate on "
                "unprivileged instances - false negative rate on privileged "
                "instances): {}".format(
                    self.metric.false_negative_rate_difference()))

    def false_negative_rate_ratio(self):
        return ("False negative rate ratio (false negative rate on "
                "unprivileged instances / false negative rate on privileged "
                "instances): {}".format(
                    self.metric.false_negative_rate_ratio()))

    def false_omission_rate(self, privileged=None):
        if privileged is None:
            return "False omission rate (FOR = FN / (FN + TN)): {}".format(
                self.metric.false_omission_rate(privileged=privileged))
        return "False omission rate on {} instances: {}".format(
            'privileged' if privileged else 'unprivileged',
            self.metric.false_omission_rate(privileged=privileged))

    def falses_omission_rate_difference(self):
        return ("False omission rate difference (falses omission rate on "
                "unprivileged instances - falses omission rate on privileged "
                "instances): {}".format(
                    self.metric.falses_omission_rate_difference()))

    def false_omission_rate_ratio(self):
        return ("False omission rate ratio (false omission rate on "
                "unprivileged instances - false omission rate on privileged "
                "instances): {}".format(
                    self.metric.false_omission_rate_ratio()))

    def false_positive_rate(self, privileged=None):
        if privileged is None:
            return "False positive rate (FPR = FP / (FP + TN)): {}".format(
                self.metric.false_positive_rate(privileged=privileged))
        return "False positive rate on {} instances: {}".format(
            'privileged' if privileged else 'unprivileged',
            self.metric.false_positive_rate(privileged=privileged))

    def false_positive_rate_difference(self):
        return ("False positive rate difference (false positive rate on "
                "unprivileged instances - false positive rate on privileged "
                "instances): {}".format(
                    self.metric.false_positive_rate_difference()))

    def false_positive_rate_ratio(self):
        return ("False positive rate ratio (false positive rate on "
                "unprivileged instances / false positive rate on privileged "
                "instances): {}".format(
                    self.metric.false_positive_rate_ratio()))

    def generalized_entropy_index(self, alpha=2):
        return "Generalized entropy index (GE(alpha)): {}".format(
            self.metric.generalized_entropy_index(alpha=alpha))

    def mean_difference(self):
        return ("Mean difference (mean label value on unprivileged instances - "
                "mean label value on privileged instances): {}".format(
                    self.metric.mean_difference()))

    def negative_predictive_value(self, privileged=None):
        if privileged is None:
            return "Negative predictive value (NPV = TN / (TN + FN)): {}".format(
                self.metric.negative_predictive_value(privileged=privileged))
        return "Negative predictive value on {} instances: {}".format(
            'privileged' if privileged else 'unprivileged',
            self.metric.negative_predictive_value(privileged=privileged))

    def num_false_negatives(self, privileged=None):
        if privileged is None:
            return "Number of false negative instances (FN): {}".format(
                self.metric.num_false_negatives(privileged=privileged))
        return "Number of {} false negative instances: {}".format(
            'privileged' if privileged else 'unprivileged',
            self.metric.num_false_negatives(privileged=privileged))

    def num_false_positives(self, privileged=None):
        if privileged is None:
            return "Number of false positive instances (FP): {}".format(
                self.metric.num_false_positives(privileged=privileged))
        return "Number of {} false positive instances: {}".format(
            'privileged' if privileged else 'unprivileged',
            self.metric.num_false_positives(privileged=privileged))

    def num_instances(self, privileged=None):
        if privileged is None:
            return "Number of instances: {}".format(
                self.metric.num_instances(privileged=privileged))
        return "Number of {} instances: {}".format(
            'privileged' if privileged else 'unprivileged',
            self.metric.num_instances(privileged=privileged))

    def num_negatives(self, privileged=None):
        if privileged is None:
            return "Number of negative-outcome instances: {}".format(
                self.metric.num_negatives(privileged=privileged))
        return "Number of {} negative-outcome instances: {}".format(
            'privileged' if privileged else 'unprivileged',
            self.metric.num_negatives(privileged=privileged))

    def num_positives(self, privileged=None):
        if privileged is None:
            return "Number of positive-outcome instances: {}".format(
                self.metric.num_positives(privileged=privileged))
        return "Number of {} positive-outcome instances: {}".format(
            'privileged' if privileged else 'unprivileged',
            self.metric.num_positives(privileged=privileged))

    def num_pred_negatives(self, privileged=None):
        if privileged is None:
            return "Number of negative-outcome instances predicted: {}".format(
                self.metric.num_pred_negatives(privileged=privileged))
        return "Number of {} negative-outcome instances predicted: {}".format(
            'privileged' if privileged else 'unprivileged',
            self.metric.num_pred_negatives(privileged=privileged))

    def num_pred_positives(self, privileged=None):
        if privileged is None:
            return "Number of positive-outcome instances predicted: {}".format(
                self.metric.num_pred_positives(privileged=privileged))
        return "Number of {} positive-outcome instances predicted: {}".format(
            'privileged' if privileged else 'unprivileged',
            self.metric.num_pred_positives(privileged=privileged))

    def num_true_negatives(self, privileged=None):
        if privileged is None:
            return "Number of true negative instances (TN): {}".format(
                self.metric.num_true_negatives(privileged=privileged))
        return "Number of {} true negative instances: {}".format(
            'privileged' if privileged else 'unprivileged',
            self.metric.num_true_negatives(privileged=privileged))

    def num_true_positives(self, privileged=None):
        if privileged is None:
            return "Number of true positive instances (TP): {}".format(
                self.metric.num_true_positives(privileged=privileged))
        return "Number of {} true positive instances: {}".format(
            'privileged' if privileged else 'unprivileged',
            self.metric.num_true_positives(privileged=privileged))

    def positive_predictive_value(self, privileged=None):
        if privileged is None:
            return "Positive predictive value (PPV, precision = TP / (TP + FP)): {}".format(
                self.metric.positive_predictive_value(privileged=privileged))
        return "Positive predictive value on {} instances: {}".format(
            'privileged' if privileged else 'unprivileged',
            self.metric.positive_predictive_value(privileged=privileged))

    def statistical_parity_difference(self):
        return ("Statistical parity difference (probability of favorable "
                "outcome for unprivileged instances - probability of favorable "
                "outcome for privileged instances): {}".format(
                    self.metric.statistical_parity_difference()))

    def theil_index(self):
        return "Theil index (generalized entropy index with alpha = 1): {}".format(
            self.metric.theil_index())

    def true_negative_rate(self, privileged=None):
        if privileged is None:
            return "True negative rate (TNR, specificity = TN / (FP + TN)): {}".format(
                self.metric.true_negative_rate(privileged=privileged))
        return "True negative rate on {} instances: {}".format(
            'privileged' if privileged else 'unprivileged',
            self.metric.true_negative_rate(privileged=privileged))

    def true_positive_rate(self, privileged=None):
        if privileged is None:
            return "True positive rate (TPR, recall, sensitivity = TP / (TP + FN)): {}".format(
                self.metric.true_positive_rate(privileged=privileged))
        return "True positive rate on {} instances: {}".format(
            'privileged' if privileged else 'unprivileged',
            self.metric.true_positive_rate(privileged=privileged))

    def true_positive_rate_difference(self):
        return ("True positive rate difference (true positive rate on "
                "unprivileged instances - true positive rate on privileged "
                "instances): {}".format(
                    self.metric.true_positive_rate_difference()))

    # ============================== ALIASES ===================================
    def equal_opportunity_difference(self):
        return self.true_positive_rate_difference()

    def power(self, privileged=None):
        return self.num_true_positives(privileged=privileged)

    def precision(self, privileged=None):
        return self.positive_predictive_value(privileged=privileged)

    def recall(self, privileged=None):
        return self.true_positive_rate(privileged=privileged)

    def sensitivity(self, privileged=None):
        return self.true_positive_rate(privileged=privileged)

    def specificity(self, privileged=None):
        return self.true_negative_rate(privileged=privileged)
