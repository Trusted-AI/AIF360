
<!-- README.md is generated from README.Rmd. Please edit that file -->

# AI Fairness 360 (AIF360) R Package

<!-- badges: start -->

[![CRAN\_Status\_Badge](http://www.r-pkg.org/badges/version/aif360)](https://cran.r-project.org/package=aif360)
<!-- badges: end -->

## Overview

The AI Fairness 360 toolkit is an open-source library to help detect and
mitigate bias in machine learning models. The AI Fairness 360 R package
includes a comprehensive set of metrics for datasets and models to test
for biases, explanations for these metrics, and algorithms to mitigate
bias in datasets and models.

## Installation

Install the CRAN version:

``` r
install.packages("aif360")
```

Or install the development version from GitHub:

``` r
# install.packages("devtools")
devtools::install_github("IBM/AIF360/aif360/aif360-r") 
```

Then, use the install\_aif360() function to install AIF360:

``` r
library(aif360)
install_aif360()
```

## Getting Started

``` r
load_aif360_lib()
```

``` r
# load a toy dataset
data <- data.frame("feature1" = c(0,0,1,1,1,1,0,1,1,0), 
                   "feature2" = c(0,1,0,1,1,0,0,0,0,1),
+                  "label" = c(1,0,0,1,0,0,1,0,1,1))
# format the dataset
formatted_dataset <- aif360::aif_dataset(data_path = data, 
                                          favor_label = 0, 
                                          unfavor_label = 1, 
                                          unprivileged_protected_attribute = 0, 
                                          privileged_protected_attribute = 1, 
                                          target_column = "label", 
                                          protected_attribute = "feature1")
```

## Contributing

If you’d like to contribute to the development of aif360, please read
[these guidelines](CONTRIBUTING.md).

Please note that the aif360 project is released with a [Contributor Code
of Conduct](CODEOFCONDUCT.md). By contributing to this project, you
agree to abide by its terms.
