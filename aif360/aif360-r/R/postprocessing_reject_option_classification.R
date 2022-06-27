#' Reject option classification
#'
#' @description Reject option classification  is a postprocessing technique that gives
#' favorable outcomes to unpriviliged groups and unfavorable outcomes to
#' priviliged groups in a confidence band around the decision boundary with
#' the highest uncertainty.
#' @param unprivileged_groups A list epresentation for unprivileged group.
#' @param privileged_groups A list representation for privileged group.
#' @param low_class_thresh Smallest classification threshold to use in the optimization. Should be between 0. and 1.
#' @param high_class_thresh Highest classification threshold to use in the optimization. Should be between 0. and 1.
#' @param num_class_thresh Number of classification thresholds between low_class_thresh and high_class_thresh for the optimization search. Should be > 0.
#' @param num_ROC_margin Number of relevant ROC margins to be used in the optimization search. Should be > 0.
#' @param metric_name Name of the metric to use for the optimization. Allowed options are "Statistical parity difference", "Average odds difference", "Equal opportunity difference".
#' @param metric_ub Upper bound of constraint on the metric value
#' @param metric_lb Lower bound of constraint on the metric value
#' @examples
#' \dontrun{
#' # Example with Adult Dataset
#' load_aif360_lib()
#' ad <- adult_dataset()
#' p <- list("race",1)
#' u <- list("race", 0)
#'
#' col_names <- c(ad$feature_names, "label")
#' ad_df <- data.frame(ad$features, ad$labels)
#' colnames(ad_df) <- col_names
#'
#' lr <- glm(label ~ ., data=ad_df, family=binomial)
#'
#' ad_prob <- predict(lr, ad_df)
#' ad_pred <- factor(ifelse(ad_prob> 0.5,1,0))
#'
#' ad_df_pred <- data.frame(ad_df)
#' ad_df_pred$label <- as.character(ad_pred)
#' colnames(ad_df_pred) <- c(ad$feature_names, 'label')
#'
#' ad_ds <- binary_label_dataset(ad_df, target_column='label', favor_label = 1,
#'                      unfavor_label = 0, unprivileged_protected_attribute = 0,
#'                      privileged_protected_attribute = 1, protected_attribute='race')
#'
#' ad_ds_pred <- binary_label_dataset(ad_df_pred, target_column='label', favor_label = 1,
#'                unfavor_label = 0, unprivileged_protected_attribute = 0,
#'                privileged_protected_attribute = 1, protected_attribute='race')
#'
#' roc <- reject_option_classification(unprivileged_groups = u,
#'                                    privileged_groups = p,
#'                                    low_class_thresh = 0.01,
#'                                    high_class_thresh = 0.99,
#'                                    num_class_thresh = as.integer(100),
#'                                    num_ROC_margin = as.integer(50),
#'                                    metric_name = "Statistical parity difference",
#'                                    metric_ub = 0.05,
#'                                    metric_lb = -0.05)
#'
#' roc <- roc$fit(ad_ds, ad_ds_pred)
#'
#' ds_transformed_pred <- roc$predict(ad_ds_pred)
#' }
#' @export
#'
reject_option_classification <- function(unprivileged_groups,
                                         privileged_groups,
                                         low_class_thresh=0.01,
                                         high_class_thresh=0.99,
                                         num_class_thresh=as.integer(100),
                                         num_ROC_margin=as.integer(50),
                                         metric_name='Statistical parity difference',
                                         metric_ub=0.05,
                                         metric_lb=-0.05){

  u_dict <- dict_fn(unprivileged_groups)
  p_dict <- dict_fn(privileged_groups)

  return(post_algo$RejectOptionClassification(u_dict,
                                              p_dict,
                                              low_class_thresh,
                                              high_class_thresh,
                                              num_class_thresh,
                                              num_ROC_margin,
                                              metric_name,
                                              metric_ub,
                                              metric_lb))
}


