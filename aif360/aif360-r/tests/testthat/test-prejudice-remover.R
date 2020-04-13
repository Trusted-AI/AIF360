context("Comprehensive Test for Prejudice Remover Algorithm")


test_that("running prejudice remover test", {
  dd <- aif360::aif_dataset(
    data_path = system.file("extdata", "preprocessed_data.csv", package="aif360"),
    favor_label=0,
    unfavor_label=1,
    unprivileged_protected_attribute=1,
    privileged_protected_attribute=0,
    target_column="Attrition",
    protected_attribute="OverTime")

  bm <- binary_label_dataset_metric(dd, list('OverTime', 0), list('OverTime',1))

  expect_equal(bm$mean_difference(), -0.203, tolerance=0.001)

  u = list('OverTime', 0)
  p = list('OverTime',1)

  model = prejudice_remover(class_attr = "Attrition", sensitive_attr = "OverTime")

  model$fit(dd)

  dd_pred <- model$predict(dd)

})

