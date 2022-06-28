#' Adult Census Income Dataset
#' @export
adult_dataset <- function(){
  return (datasets$AdultDataset())
}

#' Bank Dataset
#' @export
bank_dataset <- function(){
  return (datasets$BankDataset())
}

#' Compas Dataset
#' @export
compas_dataset <- function(){
  return (datasets$CompasDataset())
}

#' German Dataset
#' @export
german_dataset <- function(){
  return (datasets$GermanDataset())
}

#' Law School GPA Dataset
#'@seealso
#' \href{https://aif360.readthedocs.io/en/latest/modules/generated/aif360.datasets.LawSchoolGPADataset.html#aif360.datasets.LawSchoolGPADataset}{More about the Law School GPA dataset.}
#' @export
law_school_gpa_dataset <- function(){
  return (datasets$LawSchoolGPADataset())
}
