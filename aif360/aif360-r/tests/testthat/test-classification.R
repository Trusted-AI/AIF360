context("Comprehensive Test for Classification Metric")


test_that("running dataset test", {

  act <- aif360::binary_label_dataset(
    data_path = system.file("extdata", "actual_data.csv", package="aif360"),
    favor_label=1,
    unfavor_label=0,
    unprivileged_protected_attribute=0,
    privileged_protected_attribute=1,
    target_column="income",
    protected_attribute="sex")

  pred <- aif360::binary_label_dataset(
    data_path = system.file("extdata", "predicted_data.csv", package="aif360"),
    favor_label=1,
    unfavor_label=0,
    unprivileged_protected_attribute=0,
    privileged_protected_attribute=1,
    target_column="income",
    protected_attribute="sex")

  cm <- classification_metric(act, pred, unprivileged_groups = list('sex', 0), privileged_groups = list('sex', 1))

  expect_equal(cm$accuracy(), 0.55)
  expect_equal(cm$accuracy(privileged=TRUE), 0.727, tolerance=0.000273)
  expect_equal(cm$accuracy(privileged=FALSE), 0.3333, tolerance=0.000333)
  expect_equal(cm$average_abs_odds_difference(), 0.4250, tolerance=0.000603)
  expect_equal(cm$average_odds_difference(), -0.07545, tolerance=5.32e-05)
  expect_equal(class(cm$binary_confusion_matrix()), "list")
  # expect_equal(cm$between_all_groups_coefficient_of_variation(), 0.0568, tolerance=3.65e-05)
  expect_equal(cm$between_all_groups_generalized_entropy_index(alpha=2), 0.00161, tolerance=3.49e-06)
  expect_equal(cm$between_all_groups_theil_index(), 0.0016, tolerance=8.24e-06)
  # expect_equal(cm$coefficient_of_variation(), 0.5685, tolerance=0.00105)
  expect_equal(cm$disparate_impact(), 1.629, tolerance=0.00063)
  expect_equal(cm$equal_opportunity_difference(), -0.5)
  expect_equal(cm$error_rate(), 0.45)
  expect_equal(cm$error_rate(privileged=TRUE), 0.27, tolerance=0.00273)
  expect_equal(cm$error_rate(privileged=FALSE), 0.66, tolerance=0.00667)
  expect_equal(cm$error_rate_difference(), 0.3939, tolerance=3.94e-05)
  expect_equal(cm$error_rate_ratio(), 2.44, tolerance=0.00444)
  expect_equal(cm$false_discovery_rate(), 0.857, tolerance=0.000143)
  expect_equal(cm$false_discovery_rate(privileged=TRUE), 0.66, tolerance=0.00667)
  expect_equal(cm$false_discovery_rate(privileged=FALSE), 1)
  expect_equal(cm$false_discovery_rate_difference(), 0.3333, tolerance=0.000333)
  expect_equal(cm$false_discovery_rate_ratio(), 1.5)
  expect_equal(cm$false_negative_rate(), 0.75)
  expect_equal(cm$false_negative_rate(privileged=TRUE), 0.5)
  expect_equal(cm$false_negative_rate(privileged=FALSE), 1)
  expect_equal(cm$false_negative_rate_difference(), 0.5)
  expect_equal(cm$false_negative_rate_ratio(), 2)
  expect_equal(cm$false_omission_rate(), 0.2310, tolerance=0.000769)
  expect_equal(cm$false_omission_rate(privileged=TRUE), 0.125)
  expect_equal(cm$false_omission_rate(privileged=FALSE), 0.4)
  expect_equal(cm$false_omission_rate_difference(), 0.275)
  expect_equal(cm$false_omission_rate_ratio(), 3.2)
  expect_equal(cm$false_positive_rate(), 0.375)
  expect_equal(cm$false_positive_rate(privileged=TRUE), 0.2215, tolerance=0.00222)
  expect_equal(cm$false_positive_rate(privileged=FALSE), 0.571, tolerance=0.000429)
  expect_equal(cm$false_positive_rate_difference(), 0.34901, tolerance=0.000206)
  expect_equal(class(cm$generalized_binary_confusion_matrix()), "list")
  expect_equal(cm$generalized_entropy_index(alpha=2), 0.16, tolerance=0.00163)
  expect_equal(cm$generalized_false_negative_rate(), 0.75)
  expect_equal(cm$generalized_false_negative_rate(privileged=TRUE), 0.5)
  expect_equal(cm$generalized_false_negative_rate(privileged=FALSE), 1)
  expect_equal(cm$generalized_false_positive_rate(), 0.375)
  expect_equal(cm$generalized_false_positive_rate(privileged=TRUE), 0.222, tolerance=0.00222)
  expect_equal(cm$generalized_false_positive_rate(privileged=FALSE), 0.571, tolerance=0.000429)
  expect_equal(cm$generalized_true_negative_rate(), 0.625)
  expect_equal(cm$generalized_true_negative_rate(privileged=TRUE), 0.778, tolerance=1)
  expect_equal(cm$generalized_true_negative_rate(privileged=FALSE), 0.428, tolerance=1)
  expect_equal(cm$generalized_true_positive_rate(), 0.25)
  expect_equal(cm$generalized_true_positive_rate(privileged=TRUE), 0.5)
  expect_equal(cm$generalized_true_positive_rate(privileged=FALSE), 0)
  expect_equal(cm$negative_predictive_value(), 0.769, tolerance=0.000231)
  expect_equal(cm$num_false_negatives(), 3.0)
  expect_equal(cm$num_false_positives(), 6.0)
  expect_equal(cm$num_generalized_false_negatives(), 3.0)
  expect_equal(cm$num_generalized_false_positives(), 6.0)
  expect_equal(cm$num_generalized_true_negatives(), 10.0)
  expect_equal(cm$num_generalized_true_positives(), 1.0)
  expect_equal(class(cm$performance_measures()), "list")
  expect_equal(cm$positive_predictive_value(), 0.143, tolerance=0.5)
  expect_equal(cm$power(), 1)
  expect_equal(cm$precision(), 0.143, tolerance=0.5)
  expect_equal(cm$recall(), 0.25)
  expect_equal(cm$selection_rate(), 0.35)
  expect_equal(cm$specificity(), 0.625)
  expect_equal(cm$sensitivity(), 0.25)
  expect_equal(cm$statistical_parity_difference(), 0.172, tolerance=0.5)
  expect_equal(cm$theil_index(), 0.220, tolerance=0.5)
  expect_equal(cm$true_negative_rate(), 0.625)
  expect_equal(cm$true_positive_rate(), 0.25)
  expect_equal(cm$true_positive_rate_difference(), -0.5)

})
