# helper function to skip tests if we don't have the 'foo' module
skip_if_no_py_modules <- function() {
  have_scipy <- reticulate::py_module_available("aif360")
  if (!have_scipy)
    skip("AIF360 not available for testing")
}

# then call this function from all of your tests
test_that("Things work as expected", {
  skip_if_no_py_modules()
  # load aif library
  load_aif360_lib()
})


