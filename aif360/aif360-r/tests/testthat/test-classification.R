context("Comprehensive Test for Classification Metric")


test_that("running dataset test", {

  act <- aif360::aif_dataset(
    data_path = system.file("extdata", "actual_test.csv", package="aif360"),
    favor_label=1,
    unfavor_label=0,
    unprivileged_protected_attribute=1,
    privileged_protected_attribute=0,
    target_column="Attrition",
    protected_attribute="OverTime")

  pred <- aif360::aif_dataset(
    data_path = system.file("extdata", "predicted_test.csv", package="aif360"),
    favor_label=1,
    unfavor_label=0,
    unprivileged_protected_attribute=1,
    privileged_protected_attribute=0,
    target_column="Attrition",
    protected_attribute="OverTime")

  cm <- classification_metric(act, pred, unprivileged_groups = list('OverTime', 1), privileged_groups = list('OverTime', 0))

  expect_equal(cm$accuracy(), 0.5)
  expect_equal(cm$accuracy(privileged=TRUE), 0.16666666666666666)
  expect_equal(cm$accuracy(privileged=FALSE), 1.0)
  expect_equal(cm$average_abs_odds_difference(), NaN)
  expect_equal(cm$average_odds_difference(), NaN)
  expect_equal(class(cm$binary_confusion_matrix()), "list")
  expect_equal(cm$between_all_groups_coefficient_of_variation(), 0.3849001794597501)
  expect_equal(cm$between_all_groups_generalized_entropy_index(alpha=2), 0.03703703703703696)
  expect_equal(cm$between_all_groups_theil_index(), 0.03903448117673357)
  expect_equal(cm$coefficient_of_variation(), 0.47140452079103157)
  expect_equal(cm$disparate_impact(), 0.6)
  expect_equal(cm$equal_opportunity_difference(), NaN)
  expect_equal(cm$error_rate(), 0.5)
  expect_equal(cm$error_rate(privileged=TRUE), 0.8333333333333334)
  expect_equal(cm$error_rate(privileged=FALSE), 0.0)
  expect_equal(cm$error_rate_difference(), -0.8333333333333334)
  expect_equal(cm$error_rate_ratio(), 0)
  expect_equal(cm$false_discovery_rate(), 0.7142857142857143)
  expect_equal(cm$false_discovery_rate(privileged=TRUE), 1.0)
  expect_equal(cm$false_discovery_rate(privileged=FALSE), 0.0)
  expect_equal(cm$false_discovery_rate_difference(), -1)
  expect_equal(cm$false_discovery_rate_ratio(), 0.0)
  expect_equal(cm$false_negative_rate(), 0.0)
  expect_equal(cm$false_negative_rate(privileged=TRUE), NaN)
  expect_equal(cm$false_negative_rate(privileged=FALSE), 0.0)
  expect_equal(cm$false_negative_rate_difference(), NaN)
  expect_equal(cm$false_negative_rate_ratio(), NaN)
  expect_equal(cm$false_omission_rate(), 0.0)
  expect_equal(cm$false_omission_rate(privileged=TRUE), 0.0)
  expect_equal(cm$false_omission_rate(privileged=FALSE), 0.0)
  expect_equal(cm$false_omission_rate_difference(), 0.0)
  expect_equal(cm$false_omission_rate_ratio(), NaN)
  expect_equal(cm$false_positive_rate(), 0.625)
  expect_equal(cm$false_positive_rate(privileged=TRUE), 0.8333333333333334)
  expect_equal(cm$false_positive_rate(privileged=FALSE), 0.0)
  expect_equal(cm$false_positive_rate_difference(), -0.8333333333333334)
  expect_equal(class(cm$generalized_binary_confusion_matrix()), "list")
  expect_equal(cm$generalized_entropy_index(alpha=2), 0.055555555555555525)
  expect_equal(cm$generalized_false_negative_rate(), 0.0)
  expect_equal(cm$generalized_false_negative_rate(privileged=TRUE), NaN)
  expect_equal(cm$generalized_false_negative_rate(privileged=FALSE), 0.0)
  expect_equal(cm$generalized_false_positive_rate(), 0.625)
  expect_equal(cm$generalized_false_positive_rate(privileged=TRUE), 0.8333333333333334)
  expect_equal(cm$generalized_false_positive_rate(privileged=FALSE), 0.0)
  expect_equal(cm$generalized_true_negative_rate(), 0.375)
  expect_equal(cm$generalized_true_negative_rate(privileged=TRUE), 0.16666666666666666)
  expect_equal(cm$generalized_true_negative_rate(privileged=FALSE), 1.0)
  expect_equal(cm$generalized_true_positive_rate(), 1)
  expect_equal(cm$generalized_true_positive_rate(privileged=TRUE), NaN)
  expect_equal(cm$generalized_true_positive_rate(privileged=FALSE), 1)
  expect_equal(cm$negative_predictive_value(), 1.0)
  expect_equal(cm$num_false_negatives(), 0.0)
  expect_equal(cm$num_false_positives(), 5.0)
  expect_equal(cm$num_generalized_false_negatives(), 0.0)
  expect_equal(cm$num_generalized_false_positives(), 5.0)
  expect_equal(cm$num_generalized_true_negatives(), 3.0)
  expect_equal(cm$num_generalized_true_positives(), 2.0)
  expect_equal(class(cm$performance_measures()), "list")
  expect_equal(cm$positive_predictive_value(), 0.2857142857142857)
  expect_equal(cm$power(), 2)
  expect_equal(cm$precision(), 0.2857142857142857)
  expect_equal(cm$recall(), 1.0)
  expect_equal(cm$selection_rate(), 0.7)
  expect_equal(cm$specificity(), 0.375)
  expect_equal(cm$sensitivity(), 1.0)
  expect_equal(cm$statistical_parity_difference(), -0.33333333333333337)
  expect_equal(cm$theil_index(), 0.056633012265132454)
  expect_equal(cm$true_negative_rate(), 0.375)
  expect_equal(cm$true_positive_rate(), 1.0)
  expect_equal(cm$true_positive_rate_difference(), NaN)

})
