#' Dataset Metric
#' @description
#' Class for computing metrics based on one StructuredDataset.
#'
#' @param data A StructuredDataset.
#' @param privileged_groups  Privileged groups. List containing privileged protected attribute name and value of the privileged protected attribute.
#' @param unprivileged_groups  Unprivileged groups. List containing unprivileged protected attribute name and value of the unprivileged protected attribute.
#' @usage
#' dataset_metric(data, privileged_groups, unprivileged_groups)
#' @examples
#' \dontrun{
#' load_aif360_lib()
#' # Input dataset
#' data <- data.frame("feat" = c(0,0,1,1,1,1,0,1,1,0), "label" = c(1,0,0.5,1,0,0,0.5,0,0.5,1))
#' # Create structured dataset
#' sd <- aif360::structured_dataset(data_path = data, target_column="label",
#'                                  protected_attribute="feat",
#'                                  unprivileged_protected_attribute=0,
#'                                  privileged_protected_attribute=1
#' )
#' # Create an instance of dataset metric
#' dm <- dataset_metric(sd, list('feat', 1), list('feat',2))
#' # Access metric functions
#' dm$num_instances()
#' }
#' @seealso
#' \href{https://aif360.readthedocs.io/en/latest/modules/metrics.html#dataset-metric}{Explore available dataset metrics here}
#'
#' Available metrics:
#' \itemize{
#'   \item difference
#'   \item num_instances
#'   \item ratio
#' }
#' @export
#' @importFrom reticulate py_suppress_warnings import
#'
dataset_metric <- function(data,
                           privileged_groups,
                           unprivileged_groups){

   p_dict <- dict_fn(privileged_groups)
   u_dict <- dict_fn(unprivileged_groups)

   return(metrics$DatasetMetric(data,
                                privileged_groups = p_dict,
                                unprivileged_groups = u_dict))
}
