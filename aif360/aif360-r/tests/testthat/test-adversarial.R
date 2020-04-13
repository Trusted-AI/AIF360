context("Comphrehensive Test for Inprocessing Adversarial DebiasingAlgorithm")


test_that("running dataset test", {
  dd <- aif360::aif_dataset(
    data_path = system.file("extdata", "preprocessed_data.csv", package="aif360"),
    favor_label=0,
    unfavor_label=1,
    unprivileged_protected_attribute=1,
    privileged_protected_attribute=0,
    target_column="Attrition",
    protected_attribute="OverTime")

  pred <- aif360::aif_dataset(
    data_path = system.file("extdata", "actual_test.csv", package="aif360"),
    favor_label=1,
    unfavor_label=0,
    unprivileged_protected_attribute=1,
    privileged_protected_attribute=0,
    target_column="Attrition",
    protected_attribute="OverTime")

  expect_equal(dd$favorable_label, 0)
  expect_equal(dd$unfavorable_label, 1)
  expect_equal(class(dd$labels), "matrix")

  bm <- binary_label_dataset_metric(dd, list('OverTime', 0), list('OverTime',1))

  expect_equal(bm$mean_difference(), -0.2025797, tolerance=2e-08)

  ad <- adversarial_debiasing(unprivileged_groups = list('OverTime', 1), privileged_groups = list('OverTime', 0))

  model <- ad$fit(dd)

  model$predict(pred)

})
