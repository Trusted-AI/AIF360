context("Comprehensive Test for Structured Dataset Creation")


test_that("running structured dataset test", {

  sd <- aif360::structured_dataset(
    data_path = system.file("extdata", "data.csv", package="aif360"),
    target_column="income",
    protected_attribute="sex",
    unprivileged_protected_attribute=0,
    privileged_protected_attribute=1
)

  expect_equal(sd$label_names, "income")
  expect_equal(sd$protected_attribute_names, "sex")
  expect_equal(sd$unprivileged_protected_attributes, list(0))
  expect_equal(sd$privileged_protected_attributes, list(1))

})
