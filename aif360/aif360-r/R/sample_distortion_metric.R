#' Sample Distortion Metric
#' @description Class for computing metrics based on two StructuredDatasets.
#' @param dataset (StructuredDataset) A StructuredDataset
#' @param distorted_dataset (StructuredDataset) A StructuredDataset
#' @param privileged_groups Privileged groups. List containing privileged
#' protected attribute name and value of the privileged protected attribute.
#' @param unprivileged_groups Unprivileged groups. List containing unprivileged
#' protected attribute name and value of the unprivileged protected attribute.
#' @usage
#' sample_distortion_metric(dataset, distorted_dataset,
#'                          privileged_groups, unprivileged_groups)
#' @examples
#' \dontrun{
#' load_aif360_lib()
#' # Input dataset
#' features <- c(0,0,1,1,1,1,0,1,1,0)
#' labels <- c(1,0,0,1,0,0,1,0,1,1)
#' data <- data.frame("feat" = features, "label" = labels)
#' # Create aif compatible input dataset
#' act <- aif360::structured_dataset(data_path = data, target_column="label",
#'                                   protected_attribute="feat",
#'                                   unprivileged_protected_attribute=0,
#'                                   privileged_protected_attribute=1
#' )
#' # Distorted dataset
#' distorted_features <- features + 1
#' # Create aif compatible distorted dataset
#' dist <- act$copy(TRUE)
#' dist$features <- as.matrix(distorted_features)
#' # Create an instance of sample distortion metric
#' sdm <- aif360::sample_distortion_metric(act, dist, list('feat', 1), list('feat', 0))
#' # Access metric functions
#' sdm$total_euclidean_distance()
#' }
#'
#' @seealso
#' \href{https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.SampleDistortionMetric.html}{Explore available sample distortion metrics and explanations here}
#'
#' Available metrics:
#' \itemize{
#'   \item average
#'   \item average_euclidean_distance
#'   \item average_mahalanobis_distance
#'   \item average_manhattan_distance
#'   \item difference
#'   \item euclidean_distance
#'   \item mahalanobis_distance
#'   \item manhattan_distance
#'   \item maximum
#'   \item maximum_euclidean_distance
#'   \item maximum_mahalanobis_distance
#'   \item maximum_manhattan_distance
#'   \item mean_euclidean_distance_difference
#'   \item mean_euclidean_distance_ratio
#'   \item mean_mahalanobis_distance_difference
#'   \item mean_mahalanobis_distance_ratio
#'   \item mean_manhattan_distance_difference
#'   \item mean_manhattan_distance_ratio
#'   \item num_instances
#'   \item ratio
#'   \item total
#'   \item total_euclidean_distance
#'   \item total_mahalanobis_distance
#'   \item total_manhattan_distance
#'
#' }
#' @export
#' @importFrom reticulate py_suppress_warnings

sample_distortion_metric <- function(dataset,
                                     distorted_dataset,
                                     privileged_groups,
                                     unprivileged_groups
                                     ) {

  p_dict <- dict_fn(privileged_groups)
  u_dict <- dict_fn(unprivileged_groups)

  return(metrics$SampleDistortionMetric(dataset,
                                        distorted_dataset,
                                        privileged_groups = p_dict,
                                        unprivileged_groups = u_dict))
}
