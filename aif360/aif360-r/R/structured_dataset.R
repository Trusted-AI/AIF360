#' AIF360 Structured Dataset
#' @description
#' Function to create AIF structured dataset.
#' @param data_path Path to the input CSV file or a R dataframe.
#' @param target_column Name describing the label.
#' @param protected_attribute A feature for which fairness is desired.
#' @param unprivileged_protected_attribute An unprotected attribute value which is considered privileged from a fairness perspective.
#' @param privileged_protected_attribute A protected attribute value which is considered privileged from a fairness perspective.

#' @usage
#' structured_dataset(data_path, target_column, protected_attribute,
#'                      unprivileged_protected_attribute,
#'                      privileged_protected_attribute)
#' @examples
#' \dontrun{
#' load_aif360_lib()
#' # Input dataset
#' data <- data.frame("feat" = c(0,0,1,1,1,1,0,1,1,0), "label" = c(1,0,0.5,1,0,0,0.5,0,0.5,1))
#' # Create aif structured dataset
#' act <- aif360::structured_dataset(data_path = data, target_column="label",
#'                                   protected_attribute="feat",
#'                                   unprivileged_protected_attribute=0,
#'                                   privileged_protected_attribute=1
#'                                  )
#' }
#' @seealso
#' \href{https://aif360.readthedocs.io/en/latest/modules/generated/aif360.datasets.StructuredDataset.html#aif360.datasets.StructuredDataset}{More about AIF structured dataset.}
#' @export
#' @importFrom reticulate py_suppress_warnings py_dict r_to_py
#' @importFrom utils file_test

structured_dataset <- function(data_path, target_column, protected_attribute,
                               unprivileged_protected_attribute,
                               privileged_protected_attribute
                               ) {

  if (is.data.frame(data_path)) {
    dataframe <- r_to_py(data_path)
  } else if (file_test("-f", data_path) == TRUE) {
    dataframe = input_data(data_path)
  }

  target_column_list <- list_fn(target_column)
  protected_attribute_list <- list_fn(protected_attribute)
  unprivileged_protected_list <- list_of_list(unprivileged_protected_attribute)
  privileged_protected_list <- list_of_list(privileged_protected_attribute)

  return(datasets$StructuredDataset(df = dataframe,
                                    label_names = target_column_list,
                                    protected_attribute_names = protected_attribute_list,
                                    unprivileged_protected_attributes = unprivileged_protected_list,
                                    privileged_protected_attributes = privileged_protected_list
                                    ))
}
