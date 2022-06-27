context("Comprehensive Test for Disparate Impact Remover Algorithm")


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
  expect_equal(bm$disparate_impact(), 1.28, tolerance=0.00296)

  dr <- disparate_impact_remover(repair_level=1.0, sensitive_attribute='sex')
  new_dd <- dr$fit_transform(dd)
  new_bm <- binary_label_dataset_metric(new_dd, list('sex', 1), list('sex',0))

  expect_equal(new_bm$disparate_impact(), 1.28, tolerance=0.00296)

})
