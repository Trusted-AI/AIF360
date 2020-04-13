#' Reweighing
#' @description  Reweighing is a preprocessing technique that weights the examples in each (group, label) combination differently to ensure fairness before classification
#' @param unprivileged_groups a list with two values: the column of the protected class and the value indicating representation for unprivileged group
#' @param privileged_groups  a list with two values: the column of the protected class and the value indicating representation for privileged group
#' @usage reweighing(unprivileged_groups, privileged_groups)
#' @examples
#' \dontrun{
#' # An example using the Adult Dataset
#' load_aif360_lib()
#' ad <- adult_dataset()
#' p <- list("race", 1)
#' u <- list("race", 0)
#' rw <- reweighing(u,p)
#' rw$fit(ad)
#' ad_transformed <- rw$transform(ad)
#' ad_fit_transformed <- rw$fit_transform(ad)
#' }
#' @export
#'
reweighing <- function(unprivileged_groups, privileged_groups) {
  unprivileged_dict <- dict_fn(unprivileged_groups)
  privileged_dict <- dict_fn(privileged_groups)
  rw <- pre_algo$Reweighing(unprivileged_dict, privileged_dict)
  return (rw)
}
