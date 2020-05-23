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
