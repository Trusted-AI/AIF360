#' load functions
#' @export
#'
load_aif360_lib <- function() {
  e <- globalenv()
  e$datasets <- import("aif360.datasets")
  e$metrics <- import("aif360.metrics")
  e$pre_algo <- import("aif360.algorithms.preprocessing")
  e$in_algo <- import("aif360.algorithms.inprocessing")
  e$post_algo <- import("aif360.algorithms.postprocessing")
  e$tf <- import("tensorflow")
  lockBinding("datasets", e)
  lockBinding("metrics", e)
  lockBinding("pre_algo", e)
  lockBinding("in_algo", e)
  lockBinding("post_algo", e)
  lockBinding("tf", e)
}
