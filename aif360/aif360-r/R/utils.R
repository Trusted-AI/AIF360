#' Install aif360 and its dependencies
#'
#' @inheritParams reticulate::conda_list
#'
#' @param method Installation method. By default, "auto" automatically finds a
#'   method that will work in the local environment. Change the default to force
#'   a specific installation method. Note that the "virtualenv" method is not
#'   available on Windows. Note also
#'   that since this command runs without privilege the "system" method is
#'   available only on Windows.
#'
#' @param version AIF360 version to install. Specify "default" to install
#'  the latest release.
#'
#' @param envname Name of Python environment to install within
#'
#' @param extra_packages Additional Python packages to install.
#'
#' @param restart_session Restart R session after installing (note this will
#'   only occur within RStudio).
#'
#' @param conda_python_version the python version installed in the created conda
#'   environment. Python 3.11 is installed by default.
#'
#' @param ... other arguments passed to [reticulate::conda_install()] or
#'   [reticulate::virtualenv_install()].
#'
#'
#' @export
install_aif360 <- function(method = c("auto", "virtualenv", "conda"),
                               conda = "auto",
                               version = "default",
                               envname = NULL,
                               extra_packages = NULL,
                               restart_session = TRUE,
                               conda_python_version = "3.11",
                               ...) {

  method <- match.arg(method)

  reticulate::py_install(
    packages       = c("aif360", "numba", "BlackBoxAuditing", "tensorflow>=1.13.1,<2", "pandas",
                       "fairlearn==0.4.6", "protobuf==3.20.1"),
    envname        = envname,
    method         = method,
    conda          = conda,
    python_version = conda_python_version,
    pip            = TRUE,
    ...
  )

  cat("\nInstallation complete.\n\n")

  if (restart_session && rstudioapi::hasFun("restartSession"))
    rstudioapi::restartSession()

  invisible(NULL)
}

#' Read CSV file
#' @param inp data file
#' @noRd
#' @importFrom utils read.csv
#'
input_data <- function(inp){
  read.csv(inp)
}
#' create a list
#' @param i input for function
#' @noRd
#'
list_fn <- function(i){
  list(i)
}
#' create a list of list
#' @param i input for function
#' @noRd
#'
list_of_list <- function(i){
  list(list(i))
}
#' Create dictionary
#' @param values input
#' @noRd
#' @importFrom reticulate py_dict
#'
dict_fn <- function(values){
  c(py_dict(c(values[[1]]),c(values[[2]]), convert = FALSE))
}
