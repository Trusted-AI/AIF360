context("Comprehensive Test for Preprocessing Reweighing Algorithm")


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

  expect_equal(bm$mean_difference(), 0.196, tolerance=0.000433)

  rw <- reweighing(list('sex', 0), list('sex',1))
  new_dd <- rw$fit_transform(dd)
  new_bm <- binary_label_dataset_metric(new_dd, list('sex', 1), list('sex',0))

  expect_equal(new_bm$mean_difference(), 0)

})
