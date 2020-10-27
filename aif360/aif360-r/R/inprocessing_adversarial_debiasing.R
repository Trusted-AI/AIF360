#' Adversarial Debiasing
#' @description Adversarial debiasing is an in-processing technique that learns a classifier to maximize prediction accuracy and simultaneously reduce an adversary's ability to determine the protected attribute from the predictions
#' @param unprivileged_groups a list with two values: the column of the protected class and the value indicating representation for unprivileged group
#' @param privileged_groups a list with two values: the column of the protected class and the value indicating representation for privileged group
#' @param scope_name scope name for the tensorflow variables
#' @param sess tensorflow session
#' @param seed seed to make `predict` repeatable.
#' @param adversary_loss_weight hyperparameter that chooses the strength of the adversarial loss.
#' @param num_epochs number of training epochs.
#' @param batch_size batch size.
#' @param classifier_num_hidden_units number of hidden units in the classifier model.
#' @param debias learn a classifier with or without debiasing.
#' @examples
#' \dontrun{
#' load_aif360_lib()
#' ad <- adult_dataset()
#' p <- list("race", 1)
#' u <- list("race", 0)
#'
#' sess = tf$compat$v1$Session()
#'
#' plain_model = adversarial_debiasing(privileged_groups = p,
#'                                     unprivileged_groups = u,
#'                                     scope_name='debiased_classifier',
#'                                     debias=TRUE,
#'                                     sess=sess)
#'
#' plain_model$fit(ad)
#' ad_nodebiasing <- plain_model$predict(ad)
#' }
#' @export
#'
adversarial_debiasing <- function(unprivileged_groups, privileged_groups, scope_name='current', sess=tf$compat$v1$Session(),
                                  seed=NULL, adversary_loss_weight=0.1, num_epochs=50, batch_size=128,
                                  classifier_num_hidden_units=200, debias=TRUE) {
  unprivileged_dict <- dict_fn(unprivileged_groups)
  privileged_dict <- dict_fn(privileged_groups)
  ad <- in_algo$AdversarialDebiasing(unprivileged_dict, privileged_dict, scope_name=scope_name, sess=sess)
  return (ad)
}
