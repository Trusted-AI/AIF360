context("Comprehensive Test for Binary Label Dataset Metric")


test_that("running dataset test", {

  dd <- aif360::aif_dataset(
    data_path = system.file("extdata", "preprocessed_data.csv", package="aif360"),
    favor_label=0,
    unfavor_label=1,
    unprivileged_protected_attribute=1,
    privileged_protected_attribute=0,
    target_column="Attrition",
    protected_attribute="OverTime")

  expect_equal(dd$favorable_label, 0)
  expect_equal(dd$unfavorable_label, 1)
  expect_equal(class(dd$labels), "matrix")

  bm <- binary_label_dataset_metric(dd, list('OverTime', 0), list('OverTime',1))

  expect_equal(bm$mean_difference(), -0.2025797, tolerance=2e-08)
  expect_equal(bm$base_rate(), 0.8360914)
  expect_equal(bm$consistency()[1], 0.7910165, tolerance=1.65e-05)
  expect_equal(bm$disparate_impact(), 0.7734, tolerance=1.09e-05)
  expect_equal(bm$num_negatives(), 208)
  expect_equal(bm$num_negatives(privileged=TRUE), 96)
  expect_equal(bm$num_negatives(privileged=FALSE), 112)
  expect_equal(bm$num_positives(), 1061)
  expect_equal(bm$num_positives(privileged=TRUE), 810)
  expect_equal(bm$num_positives(privileged=FALSE), 251)
  expect_equal(bm$statistical_parity_difference(), -0.20257968000291904)
})
