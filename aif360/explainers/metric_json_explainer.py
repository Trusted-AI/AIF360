from collections import OrderedDict
import json

from aif360.explainers import MetricTextExplainer
from aif360.metrics import BinaryLabelDatasetMetric


class MetricJSONExplainer(MetricTextExplainer):
    """Class for explaining metric values in JSON format.

    These briefly explain what a metric is and/or how it is calculated unless
    it is obvious (e.g. accuracy) and print the value.

    This class contains JSON explanations for all metric values regardless of
    which subclass they appear in. This will raise an error if the metric does
    not apply (e.g. calling `true_positive_rate` if
    `type(metric) == DatasetMetric`).
    """

    def accuracy(self, privileged=None):
        outcome = super(MetricJSONExplainer, self).accuracy(privileged=privileged)
        response = OrderedDict((
            ("metric", "Accuracy"),
            ("message", outcome),
            ("numTruePositives", self.metric.num_true_positives(privileged=privileged)),
            ("numTrueNegatives", self.metric.num_true_negatives(privileged=privileged)),
            ("numPositives", self.metric.num_positives(privileged=privileged)),
            ("numNegatives", self.metric.num_negatives(privileged=privileged)),
            ("description", "Computed as (true positive count + "
                "true negative count)/(positive_count + negative_count)."),
            ("ideal", "The ideal value of this metric is 1.0")
        ))
        return json.dumps(response)

    def average_abs_odds_difference(self):
        outcome = super(MetricJSONExplainer, self).average_abs_odds_difference()
        response = OrderedDict((
            ("metric", "Average Absolute Odds Difference"),
            ("message", outcome),
            ("numFalsePositivesUnprivileged", self.metric.num_false_positives(privileged=False)),
            ("numNegativesUnprivileged", self.metric.num_negatives(privileged=False)),
            ("numTruePositivesUnprivileged", self.metric.num_true_positives(privileged=False)),
            ("numPositivesUnprivileged", self.metric.num_positives(privileged=False)),
            ("numFalsePositivesPrivileged", self.metric.num_false_positives(privileged=True)),
            ("numNegativesPrivileged", self.metric.num_negatives(privileged=True)),
            ("numTruePositivesPrivileged", self.metric.num_true_positives(privileged=True)),
            ("numPositivesPrivileged", self.metric.num_positives(privileged=True)),
            ("description", "Computed as average difference of false positive rate (false positives / actual negatives) and true positive rate (true positives / actual positives) between unprivileged and privileged groups."),
            ("ideal", "The ideal value of this metric is 0.0.  A value of < 0 implies higher benefit for the privileged group and a value > 0 implies higher benefit for the unprivileged group.")
        ))
        return json.dumps(response)

    def average_odds_difference(self):
        outcome = super(MetricJSONExplainer, self).average_odds_difference()
        response = OrderedDict((
            ("metric", "Average Odds Difference"),
            ("message", outcome),
            ("numFalsePositivesUnprivileged", self.metric.num_false_positives(privileged=False)),
            ("numNegativesUnprivileged", self.metric.num_negatives(privileged=False)),
            ("numTruePositivesUnprivileged", self.metric.num_true_positives(privileged=False)),
            ("numPositivesUnprivileged", self.metric.num_positives(privileged=False)),
            ("numFalsePositivesPrivileged", self.metric.num_false_positives(privileged=True)),
            ("numNegativesPrivileged", self.metric.num_negatives(privileged=True)),
            ("numTruePositivesPrivileged", self.metric.num_true_positives(privileged=True)),
            ("numPositivesPrivileged", self.metric.num_positives(privileged=True)),
            ("description", "Computed as average difference of false positive rate (false positives / negatives) and true positive rate (true positives / positives) between unprivileged and privileged groups."),
            ("ideal", "The ideal value of this metric is 0.  A value of < 0 implies higher benefit for the privileged group and a value > 0 implies higher benefit for the unprivileged group.")
        ))
        return json.dumps(response)

    def between_all_groups_coefficient_of_variation(self):
        outcome = super(MetricJSONExplainer, self).between_all_groups_coefficient_of_variation()
        response = OrderedDict((
            ("metric", "Between All Groups Coefficient Of Variation"),
            ("message", outcome),
            ("description", "Computed as the square root of twice the pairwise entropy between every pair of privileged and underprivileged groups with alpha = 2."),
            ("ideal", "The ideal value of this metric is 0.") #2.0"
        ))
        return json.dumps(response)

    def between_all_groups_generalized_entropy_index(self, alpha=2):
        outcome = super(MetricJSONExplainer, self).between_all_groups_generalized_entropy_index(alpha)
        response = OrderedDict((
            ("metric", "Between All Groups Generalized Entropy Index"),
            ("message", outcome),
            ("description", "Computed as the pairwise entropy between every pair of privileged and underprivileged groups."),
            ("ideal", "The ideal value of this metric is 0.0") #1.0"
        ))
        return json.dumps(response)

    def between_all_groups_theil_index(self):
        outcome = super(MetricJSONExplainer, self).between_all_groups_theil_index()
        response = OrderedDict((
            ("metric", "Between All Groups Theil Index"),
            ("message", outcome),
            ("description", "Computed as the pairwise entropy between every pair of privileged and underprivileged groups with alpha = 1."),
            ("ideal", "The ideal value of this metric is 0.0") #1.0"
        ))
        return json.dumps(response)

    def between_group_coefficient_of_variation(self):
        outcome = super(MetricJSONExplainer, self).between_group_coefficient_of_variation()
        response = OrderedDict((
            ("metric", "Between Group Coefficient Of Variation"),
            ("message", outcome),
            ("description", "Computed as the square root of twice the pairwise entropy between a given pair of privileged and underprivileged groups with alpha = 2."),
            ("ideal", "The ideal value of this metric is 0.0") #2.0"
        ))
        return json.dumps(response)

    def between_group_generalized_entropy_index(self, alpha=2):
        outcome = super(MetricJSONExplainer, self).between_group_generalized_entropy_index(alpha)
        response = OrderedDict((
            ("metric", "Between Group Generalized Entropy Index"),
            ("message", outcome),
            ("description", "Computed as the pairwise entropy between a given pair of privileged and underprivileged groups."),
            ("ideal", "The ideal value of this metric is 0.0") #1.0"
        ))
        return json.dumps(response)

    def between_group_theil_index(self):
        outcome = super(MetricJSONExplainer, self).between_group_theil_index()
        response = OrderedDict((
            ("metric", "Between Group Theil Index"),
            ("message", outcome),
            ("description", "Computed as the pairwise entropy between a given pair of privileged and underprivileged groups with alpha = 1."),
            ("ideal", "The ideal value of this metric is 0.0") #1.0"
        ))
        return json.dumps(response)

    def coefficient_of_variation(self):
        outcome = super(MetricJSONExplainer, self).coefficient_of_variation()
        response = OrderedDict((
            ("metric", "Coefficient Of Variation"),
            ("message", outcome),
            ("description", "Computed as the square root of twice the generalized entropy index with alpha = 2."),
            ("ideal", "The ideal value of this metric is 0.0") #2.0"
        ))
        return json.dumps(response)

    def consistency(self, n_neighbors=5):
        outcome = super(MetricJSONExplainer, self).consistency(n_neighbors)
        response = OrderedDict((
            ("metric", "Consistency"),
            ("message", outcome),
            ("description", "Individual fairness metric from Zemel, Rich, et al. \"Learning fair representations.\", ICML 2013. "
                            "Measures how similar the labels are for similar instances."),
            ("ideal", "The ideal value of this metric is 1.0")
        ))
        return json.dumps(response)

    def disparate_impact(self):
        outcome = super(MetricJSONExplainer, self).disparate_impact()
        response = []
        if type(self.metric) is BinaryLabelDatasetMetric:
            response = OrderedDict((
                ("metric", "Disparate Impact"),
                ("message", outcome),
                ("numPositivePredictionsUnprivileged", self.metric.num_positives(privileged=False)),
                ("numUnprivileged", self.metric.num_instances(privileged=False)),
                ("numPositivePredictionsPrivileged", self.metric.num_positives(privileged=True)),
                ("numPrivileged", self.metric.num_instances(privileged=True)),
                ("description", "Computed as the ratio of rate of favorable outcome for the unprivileged group to that of the privileged group."),
                ("ideal", "The ideal value of this metric is 1.0 A value < 1 implies higher benefit for the privileged group and a value >1 implies a higher benefit for the unprivileged group.")
            ))
        else:
            response = OrderedDict((
                ("metric", "Disparate Impact"),
                ("message", outcome),
                ("numPositivePredictionsUnprivileged", self.metric.num_pred_positives(privileged=False)),
                ("numUnprivileged", self.metric.num_instances(privileged=False)),
                ("numPositivePredictionsPrivileged", self.metric.num_pred_positives(privileged=True)),
                ("numPrivileged", self.metric.num_instances(privileged=True)),
                ("description", "Computed as the ratio of likelihood of favorable outcome for the unprivileged group to that of the privileged group."),
                ("ideal", "The ideal value of this metric is 1.0")
            ))
        return json.dumps(response)

    def error_rate(self, privileged=None):
        outcome = super(MetricJSONExplainer, self).error_rate(privileged=privileged)
        response = OrderedDict((
            ("metric", "Error Rate"),
            ("message", outcome),
            ("numTruePositives", self.metric.num_true_positives(privileged=privileged)),
            ("numTrueNegatives", self.metric.num_true_negatives(privileged=privileged)),
            ("numPositives", self.metric.num_positives(privileged=privileged)),
            ("numNegatives", self.metric.num_negatives(privileged=privileged)),
            ("description", "Computed as  (1 -(true positive count + true negative count)/(positive_count + negative_count)). "),
            ("ideal", "The ideal value of this metric is 0.0")
        ))
        return json.dumps(response)

    def error_rate_difference(self):
        outcome = super(MetricJSONExplainer, self).error_rate_difference()
        response = OrderedDict((
            ("metric", "Error Rate Difference"),
            ("message", outcome),
            ("numTruePositivesUnprivileged", self.metric.num_true_positives(privileged=False)),
            ("numTrueNegativesUnprivileged", self.metric.num_true_negatives(privileged=False)),
            ("numPositivesUnprivileged", self.metric.num_positives(privileged=False)),
            ("numNegativesUnprivileged", self.metric.num_negatives(privileged=False)),
            ("numTruePositivesPrivileged", self.metric.num_true_positives(privileged=True)),
            ("numTrueNegativesPrivileged", self.metric.num_true_negatives(privileged=True)),
            ("numPositivePrivileged", self.metric.num_positives(privileged=True)),
            ("numNegativePrivileged", self.metric.num_negatives(privileged=True)),
            ("description", "Error rate = 1 -(true positive count + true negative count)/(positive_count + negative_count). "
                "This metric is computed as the difference of error rates between unprivileged and privileged groups."),
            ("ideal", "The ideal value of this metric is 0.0")
        ))
        return json.dumps(response)

    def error_rate_ratio(self):
        outcome = super(MetricJSONExplainer, self).error_rate_ratio()
        response = OrderedDict((
            ("metric", "Error Rate Ratio"),
            ("message", outcome),
            ("numTruePositivesUnprivileged", self.metric.num_true_positives(privileged=False)),
            ("numTrueNegativesUnprivileged", self.metric.num_true_negatives(privileged=False)),
            ("numPositivesUnprivileged", self.metric.num_positives(privileged=False)),
            ("numNegativesUnprivileged", self.metric.num_negatives(privileged=False)),
            ("numTruePositivesPrivileged", self.metric.num_true_positives(privileged=True)),
            ("numTrueNegativesPrivileged", self.metric.num_true_negatives(privileged=True)),
            ("numPositivePrivileged", self.metric.num_positives(privileged=True)),
            ("numNegativePrivileged", self.metric.num_negatives(privileged=True)),
            ("description", "Error rate = 1 -(true positive count + true negative count)/(positive_count + negative_count). "
                "This metric is computed as the ratio of error rates between unprivileged and privileged groups."),
            ("ideal", "The ideal value of this metric is 1.0")
        ))
        return json.dumps(response)

    def false_discovery_rate(self, privileged=None):
        outcome = super(MetricJSONExplainer, self).false_discovery_rate(privileged=privileged)
        response = OrderedDict((
            ("metric", "False Discovery Rate"),
            ("message", outcome),
            ("numTruePositives", self.metric.num_true_positives(privileged=privileged)),
            ("numFalsePositives", self.metric.num_false_positives(privileged=privileged)),
            ("description", "Computed as  (false positive count / (true positive count + false positive count))."),
            ("ideal", "The ideal value of this metric is 0.0")
        ))
        return json.dumps(response)

    def false_discovery_rate_difference(self):
        outcome = super(MetricJSONExplainer, self).false_discovery_rate_difference()
        response = OrderedDict((
            ("metric", "False Discovery Rate Difference"),
            ("message", outcome),
            ("numTruePositivesUnprivileged", self.metric.num_true_positives(privileged=False)),
            ("numFalsePositivesUnprivileged", self.metric.num_false_positives(privileged=False)),
            ("numTruePositivesPrivileged", self.metric.num_true_positives(privileged=True)),
            ("numFalsePositivesPrivileged", self.metric.num_false_positives(privileged=True)),
            ("description", "False discovery rate is computed as  (false positive count / (true positive count + false positive count)). "
                "This metric is computed as the difference of false discovery rate of unprivileged and privileged instances."),
            ("ideal", "The ideal value of this metric is 0.0")
        ))
        return json.dumps(response)

    def false_discovery_rate_ratio(self):
        outcome = super(MetricJSONExplainer, self).false_discovery_rate_ratio()
        response = OrderedDict((
            ("metric", "False Discovery Rate Ratio"),
            ("message", outcome),
            ("numTruePositivesUnprivileged", self.metric.num_true_positives(privileged=False)),
            ("numFalsePositivesUnprivileged", self.metric.num_false_positives(privileged=False)),
            ("numTruePositivesPrivileged", self.metric.num_true_positives(privileged=True)),
            ("numFalsePositivesPrivileged", self.metric.num_false_positives(privileged=True)),
            ("description", "False discovery rate is computed as  (false positive count / (true positive count + false positive count)). "
                "This metric is computed as the ratio of false discovery rate of unprivileged and privileged instances."),
            ("ideal", "The ideal value of this metric is 1.0")
        ))
        return json.dumps(response)

    def false_negative_rate(self, privileged=None):
        outcome = super(MetricJSONExplainer, self).false_negative_rate(privileged=privileged)
        response = OrderedDict((
            ("metric", "False Negative Rate"),
            ("message", outcome),
            ("numFalseNegatives", self.metric.num_false_negatives(privileged=privileged)),
            ("numPositives", self.metric.num_positives(privileged=privileged)),
            ("description", "Computed as  (false negagive count / total positive count)."),
            ("ideal", "The ideal value of this metric is 0.0")
        ))
        return json.dumps(response)

    def false_negative_rate_difference(self):
        outcome = super(MetricJSONExplainer, self).false_negative_rate_difference()
        response = OrderedDict((
            ("metric", "False Negative Rate Difference"),
            ("message", outcome),
            ("numFalseNegativesUnprivileged", self.metric.num_false_negatives(privileged=False)),
            ("numPositivesUnprivileged", self.metric.num_positives(privileged=False)),
            ("numFalseNegativesPrivileged", self.metric.num_false_negatives(privileged=True)),
            ("numPositivesPrivileged", self.metric.num_positives(privileged=True)),
            ("description", "False negative rate is computed as  (false negagive count / total positive count). "
                "This metric is computed as the difference of false negative rate between unprivileged and privileged instances."),
            ("ideal", "The ideal value of this metric is 0.0")
        ))
        return json.dumps(response)

    def false_negative_rate_ratio(self):
        outcome = super(MetricJSONExplainer, self).false_negative_rate_ratio()
        response = OrderedDict((
            ("metric", "False Negative Rate Ratio"),
            ("message", outcome),
            ("numFalseNegativesUnprivileged", self.metric.num_false_negatives(privileged=False)),
            ("numPositiveaUnprivileged", self.metric.num_positives(privileged=False)),
            ("numFalseNegativesPrivileged", self.metric.num_false_negatives(privileged=True)),
            ("numPositiveaPrivileged", self.metric.num_positives(privileged=True)),
            ("description", "False negative rate is computed as  (false negagive count / total positive count). "
                "This metric is computed as the ratio of false negative rate between unprivileged and privileged instances."),
            ("ideal", "The ideal value of this metric is 1.0")
        ))
        return json.dumps(response)

    def false_omission_rate(self, privileged=None):
        outcome = super(MetricJSONExplainer, self).false_omission_rate(privileged=privileged)
        response = OrderedDict((
            ("metric", "False Omission Rate"),
            ("message", outcome),
            ("numTrueNegatives", self.metric.num_true_negatives(privileged=privileged)),
            ("numFalseNegatives", self.metric.num_false_negatives(privileged=privileged)),
            ("description", "Computed as  (false negative count / (true negative count + false negative count))."),
            ("ideal", "The ideal value of this metric is 0.0")
        ))
        return json.dumps(response)

    def false_omission_rate_difference(self):
        outcome = super(MetricJSONExplainer, self).falses_omission_rate_difference()
        response = OrderedDict((
            ("metric", "False Omission Rate Difference"),
            ("message", outcome),
            ("numTrueNegativesUnprivileged", self.metric.num_true_negatives(privileged=False)),
            ("numFalseNegativesUnprivileged", self.metric.num_false_negatives(privileged=False)),
            ("numTrueNegativesPrivileged", self.metric.num_true_negatives(privileged=True)),
            ("numFalseNegativesPrivileged", self.metric.num_false_negatives(privileged=True)),
            ("description", "False omission rate is computed as  (false negative count / (true negative count + false negative count)). "
                "This metric is computed as the difference of false omission rate of underprivileged and privileged groups."),
            ("ideal", "The ideal value of this metric is 0.0")
        ))
        return json.dumps(response)

    def false_omission_rate_ratio(self):
        outcome = super(MetricJSONExplainer, self).false_omission_rate_ratio()
        response = OrderedDict((
            ("metric", "False Omission Rate Ratio"),
            ("message", outcome),
            ("numTrueNegativesUnprivileged", self.metric.num_true_negatives(privileged=False)),
            ("numFalseNegativesUnprivileged", self.metric.num_false_negatives(privileged=False)),
            ("numTrueNegativesPrivileged", self.metric.num_true_negatives(privileged=True)),
            ("numFalseNegativesPrivileged", self.metric.num_false_negatives(privileged=True)),
            ("description", "False omission rate is computed as  (false negative count / (true negative count + false negative count)). "
                "This metric is computed as the ratio of false omission rate of underprivileged and privileged groups."),
            ("ideal", "The ideal value of this metric is 1.0")
        ))
        return json.dumps(response)

    def false_positive_rate(self, privileged=None):
        outcome = super(MetricJSONExplainer, self).false_positive_rate(privileged=privileged)
        response = OrderedDict((
            ("metric", "False Positive Rate"),
            ("message", outcome),
            ("numFalsePositives", self.metric.num_false_positives(privileged=privileged)),
            ("numNegatives", self.metric.num_negatives(privileged=privileged)),
            ("description", "Computed as (false positive count / total negative count)."),
            ("ideal", "The ideal value of this metric is 0.0")
        ))
        return json.dumps(response)

    def false_positive_rate_difference(self):
        outcome = super(MetricJSONExplainer, self).false_positive_rate_difference()
        response = OrderedDict((
            ("metric", "False Positive Rate Difference"),
            ("message", outcome),
            ("numFalsePositivesUnprivileged", self.metric.num_false_positives(privileged=False)),
            ("numNegativesUnprivileged", self.metric.num_negatives(privileged=False)),
            ("numPositivesPrivileged", self.metric.num_false_positives(privileged=True)),
            ("numNegativesPrivileged", self.metric.num_negatives(privileged=True)),
            ("description", "False positive rate is computed as (false positive count / total negative count). "
                "This metric is computed as the difference of false positive rates between the unprivileged and privileged groups"),
            ("ideal", "The ideal value of this metric is 0.0")
        ))
        return json.dumps(response)

    def false_positive_rate_ratio(self):
        outcome = super(MetricJSONExplainer, self).false_positive_rate_ratio()
        response = OrderedDict((
            ("metric", "False Positive Rate Ratio"),
            ("message", outcome),
            ("numFalsePositivesUnprivileged", self.metric.num_false_positives(privileged=False)),
            ("numNegativesUnprivileged", self.metric.num_negatives(privileged=False)),
            ("numFalsePositivesPrivileged", self.metric.num_false_positives(privileged=True)),
            ("numNegativesPrivileged", self.metric.num_negatives(privileged=True)),
            ("description", "False positive rate is computed as (false positive count / total negative count). "
                "This metric is computed as the ratio of false positive rates between the unprivileged and privileged groups"),
            ("ideal", "The ideal value of this metric is 1.0")
        ))
        return json.dumps(response)

    def generalized_entropy_index(self, alpha=2):
        outcome = super(MetricJSONExplainer, self).generalized_entropy_index(alpha=alpha)
        response = OrderedDict((
            ("metric", "Generalized Entropy Index"),
            ("message", outcome),
            ("description", "This metric represents the generalized entropy index measured between the predicted and actual favorable outcomes."),
            ("ideal", "The ideal value of this metric is 0.0")
        ))
        return json.dumps(response)

    def mean_difference(self):
        outcome = super(MetricJSONExplainer, self).mean_difference()
        response = OrderedDict((
            ("metric", "Mean Difference"),
            ("message", outcome),
            ("numPositivesUnprivileged", self.metric.num_positives(privileged=False)),
            ("numInstancesUnprivileged", self.metric.num_instances(privileged=False)),
            ("numPositivesPrivileged", self.metric.num_positives(privileged=True)),
            ("numInstancesPrivileged", self.metric.num_instances(privileged=True)),
            ("description", "Computed as the difference of the rate of favorable outcomes received by the unprivileged group to the privileged group."),
            ("ideal", "The ideal value of this metric is 0.0")
        ))
        return json.dumps(response)

    def negative_predictive_value(self, privileged=None):
        outcome = super(MetricJSONExplainer, self).negative_predictive_value(privileged=privileged)
        response = OrderedDict((
            ("metric", "Negative Predictive Value"),
            ("message", outcome),
            ("numTrueNegatives", self.metric.num_true_negatives(privileged=privileged)),
            ("numFalseNegatives", self.metric.num_false_negatives(privileged=privileged)),
            ("description", "Computed as (number of true negatives / (number of true negatives + number of false negatives))."),
            ("ideal", "The ideal value of this metric is 1.0")
        ))
        return json.dumps(response)

    def num_false_negatives(self, privileged=None):
        outcome = super(MetricJSONExplainer, self).num_false_negatives(privileged=privileged)
        response = OrderedDict((
            ("metric", "Number Of False Negatives"),
            ("message", outcome),
            ("numFalseNegatives", self.metric.num_false_negatives(privileged=privileged)),
            ("description", "Computed as the number of false negative instances for the given (privileged or unprivileged) group."),
            ("ideal", "The ideal value of this metric is 0.0")
        ))
        return json.dumps(response)

    def num_false_positives(self, privileged=None):
        outcome = super(MetricJSONExplainer, self).num_false_positives(privileged=privileged)
        response = OrderedDict((
            ("metric", "Number Of False Positives"),
            ("message", outcome),
            ("numFalsePositives", self.metric.num_false_positives(privileged=privileged)),
            ("description", "Computed as the number of false positive instances for the given (privileged or unprivileged) group."),
            ("ideal", "The ideal value of this metric is 0.0")
        ))
        return json.dumps(response)

    def num_instances(self, privileged=None):
        outcome = super(MetricJSONExplainer, self).num_instances(privileged=privileged)
        response = OrderedDict((
            ("metric", "Number Of Instances"),
            ("message", outcome),
            ("numInstances", self.metric.num_instances(privileged=privileged)),
            ("description", "Computed as the number of instances for the given (privileged or unprivileged) group."),
            ("ideal", "The ideal value is the total number of instances made available")
        ))
        return json.dumps(response)

    def num_negatives(self, privileged=None):
        outcome = super(MetricJSONExplainer, self).num_negatives(privileged=privileged)
        response = OrderedDict((
            ("metric", "Number Of Negatives"),
            ("message", outcome),
            ("numNegatives", self.metric.num_negatives(privileged=privileged)),
            ("description", "Computed as the number of negative instances for the given (privileged or unprivileged) group."),
            ("ideal", "The ideal value of this metric lies in the total number of negative instances made available")
        ))
        return json.dumps(response)

    def num_positives(self, privileged=None):
        outcome = super(MetricJSONExplainer, self).num_positives(privileged=privileged)
        response = OrderedDict((
            ("metric", "Number Of Positives"),
            ("message", outcome),
            ("numPositives", self.metric.num_positives(privileged=privileged)),
            ("description", "Computed as the number of positive instances for the given (privileged or unprivileged) group."),
            ("ideal", "The ideal value of this metric lies in the total number of positive instances made available")
        ))
        return json.dumps(response)

    def num_pred_negatives(self, privileged=None):
        outcome = super(MetricJSONExplainer, self).num_pred_negatives(privileged=privileged)
        response = OrderedDict((
            ("metric", "Number Of Predicted Negatives"),
            ("message", outcome),
            ("numPredNegatives", self.metric.num_pred_negatives(privileged=privileged)),
            ("description", "Computed as the number of predicted negative instances for the given (privileged or unprivileged) group."),
            ("ideal", "The ideal value is the total number of negative instances made available")
        ))
        return json.dumps(response)

    def num_pred_positives(self, privileged=None):
        outcome = super(MetricJSONExplainer, self).num_pred_positives(privileged=privileged)
        response = OrderedDict((
            ("metric", "Number Of Predicted Positives"),
            ("message", outcome),
            ("numPredPositives", self.metric.num_pred_positives(privileged=privileged)),
            ("description", "Computed as the number of predicted positive instances for the given (privileged or unprivileged) group."),
            ("ideal", "The ideal value is the total number of positive instances made available")
        ))
        return json.dumps(response)

    def num_true_negatives(self, privileged=None):
        outcome = super(MetricJSONExplainer, self).num_true_negatives(privileged=privileged)
        response = OrderedDict((
            ("metric", "Number Of True Negatives"),
            ("message", outcome),
            ("numTrueNegatives", self.metric.num_true_negatives(privileged=privileged)),
            ("description", "Computed as the number of true negative instances for the given (privileged or unprivileged) group."),
            ("ideal", "The ideal value is the total number of negative instances made available")
        ))
        return json.dumps(response)

    def num_true_positives(self, privileged=None):
        outcome = super(MetricJSONExplainer, self).num_true_positives(privileged=privileged)
        response = OrderedDict((
            ("metric", "Number Of True Positives"),
            ("message", outcome),
            ("numTruePositives", self.metric.num_true_positives(privileged=privileged)),
            ("description", "Computed as the number of true positive instances for the given (privileged or unprivileged) group."),
            ("ideal", "The ideal value is the total number of positive instances made available")
        ))
        return json.dumps(response)

    def positive_predictive_value(self, privileged=None):
        outcome = super(MetricJSONExplainer, self).positive_predictive_value(privileged=privileged)
        response = OrderedDict((
            ("metric", "Positive Predictive Value"),
            ("message", outcome),
            ("numTruePositives", self.metric.num_true_positives(privileged=privileged)),
            ("numFalsePositives", self.metric.num_false_positives(privileged=privileged)),
            ("description", "Computed as (true positives / (true positives + false positives)) for the given (privileged or unprivileged) group."),
            ("ideal", "The ideal value is 1.0")
        ))
        return json.dumps(response)

    def statistical_parity_difference(self):
        outcome = super(MetricJSONExplainer, self).statistical_parity_difference()
        response = []
        if type(self.metric) is BinaryLabelDatasetMetric:
            response = OrderedDict((
                ("metric", "Statistical Parity Difference"),
                ("message", outcome),
                ("numPositivesUnprivileged", self.metric.num_positives(privileged=False)),
                ("numInstancesUnprivileged", self.metric.num_instances(privileged=False)),
                ("numPositivesPrivileged", self.metric.num_positives(privileged=True)),
                ("numInstancesPrivileged", self.metric.num_instances(privileged=True)),
                ("description", "Computed as the difference of the rate of favorable outcomes received by the unprivileged group to the privileged group."),
                ("ideal", " The ideal value of this metric is 0")
            ))
        else:
            response = OrderedDict((
                ("metric", "Statistical Parity Difference"),
                ("message", outcome),
                ("numPositivesUnprivileged", self.metric.num_pred_positives(privileged=False)),
                ("numInstancesUnprivileged", self.metric.num_instances(privileged=False)),
                ("numPositivesPrivileged", self.metric.num_pred_positives(privileged=True)),
                ("numInstancesPrivileged", self.metric.num_instances(privileged=True)),
                ("description", "Computed as the difference of the rate of favorable outcomes received by the unprivileged group to the privileged group."),
                ("ideal", " The ideal value of this metric is 0")
            ))
        return json.dumps(response)

    def theil_index(self):
        outcome = super(MetricJSONExplainer, self).theil_index()
        response = OrderedDict((
            ("metric", "Theil Index"),
            ("message", outcome),
            ("description", "Computed as the generalized entropy of benefit for all individuals in the dataset, with alpha = 1. It measures the inequality in benefit allocation for individuals."),
            ("ideal", "A value of 0 implies perfect fairness.")
        ))
        return json.dumps(response)

    def true_negative_rate(self, privileged=None):
        outcome = super(MetricJSONExplainer, self).true_negative_rate(privileged=privileged)
        response = OrderedDict((
            ("metric", "True Negative Rate"),
            ("message", outcome),
            ("numTrueNegatives", self.metric.num_true_negatives(privileged=privileged)),
            ("numNegatives", self.metric.num_negatives(privileged=privileged)),
            ("description", "Computed as the ratio of true negatives to the total number of negatives for the given (privileged or unprivileged) group."),
            ("ideal", "The ideal value is 1.0")
        ))
        return json.dumps(response)

    def true_positive_rate(self, privileged=None):
        outcome = super(MetricJSONExplainer, self).true_positive_rate(privileged=privileged)
        response = OrderedDict((
            ("metric", "True Positive Rate"),
            ("message", outcome),
            ("numTruePositives", self.metric.num_true_positives(privileged=privileged)),
            ("numPositives", self.metric.num_positives(privileged=privileged)),
            ("description", "Computed as the ratio of true positives to the total number of positives for the given (privileged or unprivileged) group."),
            ("ideal", "The ideal value is 1.0")
        ))
        return json.dumps(response)

    def true_positive_rate_difference(self):
        outcome = super(MetricJSONExplainer, self).true_positive_rate_difference()
        response = OrderedDict((
            ("metric", "True Positive Rate Difference"),
            ("message", outcome),
            ("numTruePositivesUnprivileged", self.metric.num_true_positives(privileged=False)),
            ("numPositivesUnprivileged", self.metric.num_positives(privileged=False)),
            ("numTruePositivesPrivileged", self.metric.num_true_positives(privileged=True)),
            ("numPositivesPrivileged", self.metric.num_positives(privileged=True)),
            ("description", "This metric is computed as the difference of true positive rates between the unprivileged and the privileged groups. "
                " The true positive rate is the ratio of true positives to the total number of actual positives for a given group."),
            ("ideal", "The ideal value is 0. A value of < 0 implies higher benefit for the privileged group and a value > 0 implies higher benefit for the unprivileged group.")
        ))
        return json.dumps(response)

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
