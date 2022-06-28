#' Adversarial Debiasing
#' @description Adversarial debiasing is an in-processing technique that learns a classifier to maximize prediction accuracy 
#' and simultaneously reduce an adversary's ability to determine the protected attribute from the predictions
#' @param unprivileged_groups A list with two values: the column of the protected class and the value indicating representation for unprivileged group.
#' @param privileged_groups A list with two values: the column of the protected class and the value indicating representation for privileged group.
#' @param scope_name Scope name for the tensorflow variables.
#' @param sess tensorflow session
#' @param seed Seed to make \code{predict} repeatable. If not, \code{NULL}, must be an integer.
#' @param adversary_loss_weight Hyperparameter that chooses the strength of the adversarial loss.
#' @param num_epochs Number of training epochs. Must be an integer.
#' @param batch_size Batch size. Must be an integer.
#' @param classifier_num_hidden_units Number of hidden units in the classifier model. Must be an integer.
#' @param debias Learn a classifier with or without debiasing.
#' @examples
#' \dontrun{
#' load_aif360_lib()
#' ad <- adult_dataset()
#' p <- list("race", 1)
#' u <- list("race", 0)
#'
#' sess <- tf$compat$v1$Session()
#'
#' plain_model <- adversarial_debiasing(privileged_groups = p,
#'                                      unprivileged_groups = u,
#'                                      scope_name = "debiased_classifier",
#'                                      debias = TRUE,
#'                                      sess = sess)
#'
#' plain_model$fit(ad)
#' ad_nodebiasing <- plain_model$predict(ad)
#' }
#' @export
#'
adversarial_debiasing <- function(unprivileged_groups, 
                                  privileged_groups, 
                                  scope_name = "current", 
                                  sess = tf$compat$v1$Session(),
                                  seed = NULL, 
                                  adversary_loss_weight = 0.1, 
                                  num_epochs = 50L, 
                                  batch_size = 128L,
                                  classifier_num_hidden_units = 200L, 
                                  debias = TRUE) {
  
  
  
  unprivileged_dict <- dict_fn(unprivileged_groups)
  privileged_dict <- dict_fn(privileged_groups)
  
  # run check for variables that must be integers
  int_vars <- list(num_epochs = num_epochs, batch_size = batch_size, classifier_num_hidden_units = classifier_num_hidden_units)
  
  if (!is.null(seed)) int_vars <- append(int_vars, c(seed = seed))
  
  is_int <- sapply(int_vars, is.integer)
  int_varnames <- names(int_vars)
  
  if (any(!is_int)) stop(paste(int_varnames[!is_int], collapse = ", "), " must be integer(s)")
  
  
  
  ad <- in_algo$AdversarialDebiasing(unprivileged_dict, 
                                     privileged_dict, 
                                     scope_name = scope_name, 
                                     sess = sess,
                                     seed = seed, 
                                     adversary_loss_weight = adversary_loss_weight,
                                     num_epochs = num_epochs,
                                     batch_size = batch_size,
                                     classifier_num_hidden_units = classifier_num_hidden_units, 
                                     debias = debias)
  return(ad)
}
