context("Comphrehensive Test for Preprocessing Reweighing Algorithm")


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

  rw <- reweighing(list('OverTime', 1), list('OverTime',0))
  new_dd <- rw$fit_transform(dd)
  new_bm <- binary_label_dataset_metric(new_dd, list('OverTime', 0), list('OverTime',1))

  expect_equal(new_bm$mean_difference(), 0)

})
