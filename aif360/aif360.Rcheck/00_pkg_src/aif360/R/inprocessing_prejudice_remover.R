#' Prejudice Remover
#' @description Prejudice remover is an in-processing technique that adds a discrimination-aware regularization term to the learning objective
#' @param eta fairness penalty parameter
#' @param sensitive_attr name of protected attribute
#' @param class_attr label name
#' @usage prejudice_remover(eta=1.0, sensitive_attr='',class_attr='')
#' @examples
#' \dontrun{
#' # An example using the Adult Dataset
#' load_aif360_lib()
#' ad <- adult_dataset()
#' model <- prejudice_remover(class_attr = "income-per-year", sensitive_attr = "race")
#' model$fit(ad)
#' ad_pred <- model$predict(ad)
#'}
#' @export
#'
prejudice_remover <- function(eta=1.0,
                              sensitive_attr='',
                              class_attr=''){

  pr <- in_algo$PrejudiceRemover(eta,
                                 sensitive_attr,
                                 class_attr)
  return(pr)
}
