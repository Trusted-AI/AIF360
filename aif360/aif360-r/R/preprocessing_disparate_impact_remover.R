#' Disparate Impact Remover
#' @description Disparate impact remover is a preprocessing technique that edits feature values increase group fairness while preserving rank-ordering within groups
#' @param repair_level Repair amount. 0.0 is no repair while 1.0 is full repair.
#' @param sensitive_attribute Single protected attribute with which to do repair.
#' @usage disparate_impact_remover(repair_level = 1.0, sensitive_attribute = '')
#' @examples
#' \dontrun{
#' # An example using the Adult Dataset
#' load_aif360_lib()
#' ad <- adult_dataset()
#' p <- list("race", 1)
#' u <- list("race", 0)
#'
#' di <- disparate_impact_remover(repair_level = 1.0, sensitive_attribute = "race")
#' rp <- di$fit_transform(ad)
#'
#' di_2 <- disparate_impact_remover(repair_level = 0.8, sensitive_attribute = "race")
#' rp_2 <- di_2$fit_transform(ad)
#' }
#' @export
#'
disparate_impact_remover <- function(repair_level=1.0, sensitive_attribute='') {
  dr <- pre_algo$DisparateImpactRemover(repair_level, sensitive_attribute)
  return (dr)
}


