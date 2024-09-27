context("Comprehensive Test for Sample Distortion Metric")

test_that("running dataset test", {

  # Create aif360 compatible dataset
  act <- aif360::structured_dataset(
    data_path = system.file("extdata", "actual_data.csv", package="aif360"),
    target_column="income",
    protected_attribute="sex",
    unprivileged_protected_attribute=0,
    privileged_protected_attribute=1
    )

  # Create aif360 compatible distorted dataset
  dist <- act$copy(TRUE)
  dist$features <- dist$features + 1

  # Create an instance of sample distortion metric
  sdm <- aif360::sample_distortion_metric(act, dist,
                                          privileged_groups = list('sex', 1),
                                          unprivileged_groups = list('sex', 0))

  # tests
  expect_equal(sdm$total_manhattan_distance(), 280.0, tolerance=0.0028)
  expect_equal(sdm$total_euclidean_distance(), 74.833, tolerance=0.0008)
  expect_equal(sdm$total_mahalanobis_distance(), 39.496, tolerance=0.0004)
  expect_equal(sdm$average_manhattan_distance(), 14.0, tolerance=0.0001)

})
