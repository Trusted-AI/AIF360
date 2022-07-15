context("Comprehensive Test for Inprocessing Adversarial DebiasingAlgorithm")


test_that("running dataset test", {
  dd <- aif360::binary_label_dataset(
    data_path = system.file("extdata", "data.csv", package="aif360"),
    favor_label=0,
    unfavor_label=1,
    unprivileged_protected_attribute=0,
    privileged_protected_attribute=1,
    target_column="income",
    protected_attribute="sex")

  pred <- aif360::binary_label_dataset(
    data_path = system.file("extdata", "actual_data.csv", package="aif360"),
    favor_label=1,
    unfavor_label=0,
    unprivileged_protected_attribute=0,
    privileged_protected_attribute=1,
    target_column="income",
    protected_attribute="sex")

  expect_equal(dd$favorable_label, 0)
  expect_equal(dd$unfavorable_label, 1)

  bm <- binary_label_dataset_metric(dd, list('sex', 1), list('sex',0))

  expect_equal(bm$mean_difference(), 0.196, tolerance=0.000433)

  ad <- adversarial_debiasing(unprivileged_groups = list('sex', 0), privileged_groups = list('sex', 1))

  model <- ad$fit(dd)

  model$predict(pred)

})
