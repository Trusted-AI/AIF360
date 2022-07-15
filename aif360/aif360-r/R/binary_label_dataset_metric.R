#' Binary Label Dataset Metric
#' @description Class for computing metrics on an aif360 compatible dataset with binary labels.
#' @param dataset A aif360 compatible dataset.
#' @param privileged_groups Privileged groups. List containing privileged protected attribute name and value of the privileged protected attribute.
#' @param unprivileged_groups Unprivileged groups. List containing unprivileged protected attribute name and value of the unprivileged protected attribute.
#' @usage
#' binary_label_dataset_metric(dataset, privileged_groups, unprivileged_groups)
#' @examples
#' \dontrun{
#' load_aif360_lib()
#' # Load the adult dataset
#' adult_dataset <- adult_dataset()
#'
#' # Define the groups
#' privileged_groups <- list("race", 1)
#' unprivileged_groups <- list("race", 0)
#'
#' # Metric for Binary Label Dataset
#' bm <- binary_label_dataset_metric(dataset = adult_dataset,
#'                                   privileged_groups = privileged_groups,
#'                                   unprivileged_groups = unprivileged_groups)
#'
#' # Difference in mean outcomes between unprivileged and privileged groups
#' bm$mean_difference()
#' }
#' @seealso
#' \href{https://aif360.readthedocs.io/en/latest/modules/metrics.html#aif360.metrics.BinaryLabelDatasetMetric}{Explore available binary label dataset metrics here}
#'
#' Available metrics are: base_rate, consistency, disparate_impact, mean_difference, num_negatives, num_positives and statistical_parity_difference.
#' @export
#' @importFrom reticulate py_suppress_warnings py_to_r
#'
binary_label_dataset_metric <- function(dataset,
                                        privileged_groups,
                                        unprivileged_groups){

  p_dict <- dict_fn(privileged_groups)
  u_dict <- dict_fn(unprivileged_groups)

  return(metrics$BinaryLabelDatasetMetric(dataset,
                                          privileged_groups = p_dict,
                                          unprivileged_groups = u_dict))
}
