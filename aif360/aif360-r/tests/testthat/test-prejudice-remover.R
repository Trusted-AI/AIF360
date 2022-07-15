context("Comprehensive Test for Prejudice Remover Algorithm")


test_that("running prejudice remover test", {
  dd <- aif360::binary_label_dataset(
    data_path = system.file("extdata", "data.csv", package="aif360"),
    favor_label=0,
    unfavor_label=1,
    unprivileged_protected_attribute=0,
    privileged_protected_attribute=1,
    target_column="income",
    protected_attribute="sex")

  bm <- binary_label_dataset_metric(dd, list('sex', 1), list('sex',0))

  expect_equal(bm$mean_difference(), 0.196, tolerance=0.000433)

  u = list('sex', 0)
  p = list('sex',1)

  model = prejudice_remover(class_attr = "income", sensitive_attr = "sex")

  model$fit(dd)

  dd_pred <- model$predict(dd)

})

