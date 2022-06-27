#' AIF360 dataset
#' @description
#' Function to create AIF compatible dataset.
#' @param data_path Path to the input CSV file or a R dataframe.
#' @param favor_label Label value which is considered favorable (i.e. “positive”).
#' @param unfavor_label Label value which is considered unfavorable (i.e. “negative”).
#' @param unprivileged_protected_attribute A unprotected attribute value which is considered privileged from a fairness perspective.
#' @param privileged_protected_attribute A protected attribute value which is considered privileged from a fairness perspective.
#' @param target_column Name describing the label.
#' @param protected_attribute A feature for which fairness is desired.
#' @usage
#' binary_label_dataset(data_path,  favor_label, unfavor_label,
#'                      unprivileged_protected_attribute,
#'                      privileged_protected_attribute,
#'                      target_column, protected_attribute)
#' @examples
#' \dontrun{
#' load_aif360_lib()
#' # Input dataset
#' data <- data.frame("feat" = c(0,0,1,1,1,1,0,1,1,0), "label" = c(1,0,0,1,0,0,1,0,1,1))
#' # Create aif compatible input dataset
#' act <- aif360::binary_label_dataset(data_path = data,  favor_label=0, unfavor_label=1,
#'                             unprivileged_protected_attribute=0,
#'                             privileged_protected_attribute=1,
#'                             target_column="label", protected_attribute="feat")
#' }
#' @seealso
#' \href{https://aif360.readthedocs.io/en/latest/modules/datasets.html#binary-label-dataset}{More about AIF binary dataset.}
#' @export
#' @importFrom reticulate py_suppress_warnings py_dict r_to_py
#' @importFrom utils file_test
#'
binary_label_dataset <- function(data_path, favor_label,
                        unfavor_label, unprivileged_protected_attribute,
                        privileged_protected_attribute,
                        target_column, protected_attribute) {

  if (is.data.frame(data_path)) {
    dataframe <- r_to_py(data_path)
  } else if (file_test("-f", data_path) == TRUE) {
    dataframe = input_data(data_path)
  }
  unprivileged_protected_list <- list_of_list(unprivileged_protected_attribute)
  privileged_protected_list <- list_of_list(privileged_protected_attribute)
  target_column_list <- list_fn(target_column)
  protected_attribute_list <- list_fn(protected_attribute)

  return(datasets$BinaryLabelDataset(df = dataframe,
                                     favorable_label = favor_label,
                                     unfavorable_label = unfavor_label,
                                     unprivileged_protected_attributes = unprivileged_protected_list,
                                     privileged_protected_attributes = privileged_protected_list,
                                     label_names = target_column_list,
                                     protected_attribute_names = protected_attribute_list))

}















