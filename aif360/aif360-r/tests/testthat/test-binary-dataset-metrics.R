context("Comprehensive Test for Binary Label Dataset Metric")


test_that("running dataset test", {

  dd <- aif360::binary_label_dataset(
    data_path = system.file("extdata", "data.csv", package="aif360"),
    favor_label=0,
    unfavor_label=1,
    unprivileged_protected_attribute=0,
    privileged_protected_attribute=1,
    target_column="income",
    protected_attribute="sex")

  expect_equal(dd$favorable_label, 0)
  expect_equal(dd$unfavorable_label, 1)

  bm <- binary_label_dataset_metric(dd, list('sex', 1), list('sex',0))

  expect_equal(bm$mean_difference(), 0.196,tolerance=0.000433)
  expect_equal(bm$base_rate(), 0.7591653606219844)
  expect_equal(bm$consistency()[1], 0.769, tolerance=0.0144)
  expect_equal(bm$disparate_impact(), 1.28, tolerance=0.00296)
  expect_equal(bm$num_negatives(), 7837)
  expect_equal(bm$num_negatives(privileged=TRUE), 6660)
  expect_equal(bm$num_negatives(privileged=FALSE), 1177)
  expect_equal(bm$num_positives(), 24704)
  expect_equal(bm$num_positives(privileged=TRUE), 15119)
  expect_equal(bm$num_positives(privileged=FALSE), 9585)
  expect_equal(bm$statistical_parity_difference(), 0.19643287553870947)

})
