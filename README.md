
<!-- README.md is generated from README.Rmd. Please edit that file -->

<img align="right" src="docs/images/cboost_hexagon.png" width="100px">

[![Build
Status](https://travis-ci.org/schalkdaniel/compboost.svg?branch=master)](https://travis-ci.org/schalkdaniel/compboost)
[![AppVeyor Build
Status](https://ci.appveyor.com/api/projects/status/github/schalkdaniel/compboost?branch=master&svg=true)](https://ci.appveyor.com/project/schalkdaniel/compboost)
[![Coverage
Status](https://coveralls.io/repos/github/schalkdaniel/compboost/badge.svg?branch=master)](https://coveralls.io/github/schalkdaniel/compboost?branch=master)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](#license)
[![status](http://joss.theoj.org/papers/94cfdbbfdfc8796c5bdb1a74ee59fcda/status.svg)](http://joss.theoj.org/papers/94cfdbbfdfc8796c5bdb1a74ee59fcda)

[Documentation](https://compboost.org) |
[Contributors](CONTRIBUTORS.md) |
[Release Notes](NEWS.md)

## compboost: Fast and Flexible Component-Wise Boosting Framework

Component-wise boosting applies the boosting framework to statistical
models, e.g., general additive models using component-wise smoothing
splines. Boosting these kinds of models maintains interpretability and
enables unbiased model selection in high dimensional feature spaces.

The `R` package `compboost` is an alternative implementation of
component-wise boosting written in `C++` to obtain high runtime
performance and full memory control. The main idea is to provide a
modular class system which can be extended without editing the source
code. Therefore, it is possible to use `R` functions as well as `C++`
functions for custom base-learners, losses, logging mechanisms or
stopping criteria.

For an introduction and overview about the functionality visit the
[project page](https://schalkdaniel.github.io/compboost/).

## Installation

#### Developer version:

``` r
devtools::install_github("schalkdaniel/compboost")
```

## Examples

This examples are rendered using <code>compboost 0.1.0</code>.

To be as flexible as possible one should use the `R6` API do define
base-learner, losses, stopping criteria, or optimizer as desired.
Another option is to use wrapper functions as described on the [project
page](https://schalkdaniel.github.io/compboost/).

``` r
library(compboost)

# Check installed version:
packageVersion("compboost")
#> [1] '0.1.0'

# Load data set with binary classification task:
data(PimaIndiansDiabetes, package = "mlbench")
# Create categorical feature:
PimaIndiansDiabetes$pregnant.cat = ifelse(PimaIndiansDiabetes$pregnant == 0, "no", "yes")

# Define Compboost object:
cboost = Compboost$new(data = PimaIndiansDiabetes, target = "diabetes", loss = LossBinomial$new())
cboost
#> Component-Wise Gradient Boosting
#> 
#> Trained on PimaIndiansDiabetes with target diabetes
#> Number of base-learners: 0
#> Learning rate: 0.05
#> Iterations: 0
#> Positive class: neg
#> 
#> LossBinomial Loss:
#> 
#>   Loss function: L(y,x) = log(1 + exp(-2yf(x))
#> 
#> 

# Add p-spline base-learner with default parameter:
cboost$addBaselearner(feature = "pressure", id = "spline", bl.factory = BaselearnerPSpline)

# Add another p-spline learner with custom parameters:
cboost$addBaselearner(feature = "age", id = "spline", bl.factory = BaselearnerPSpline, degree = 3, 
  n.knots = 10, penalty = 4, differences = 2)

# Add categorical feature (as single linear base-learner):
cboost$addBaselearner(feature = "pregnant.cat", id = "category", bl.factory = BaselearnerPolynomial,
  degree = 1, intercept = FALSE)

# Check all registered base-learner:
cboost$getBaselearnerNames()
#> [1] "pressure_spline"           "age_spline"               
#> [3] "pregnant.cat_yes_category" "pregnant.cat_no_category"

# Train model:
cboost$train(1000L, trace = 200L)
#>    1/1000: risk = 0.66
#>  200/1000: risk = 0.58
#>  400/1000: risk = 0.57
#>  600/1000: risk = 0.57
#>  800/1000: risk = 0.57
#> 1000/1000: risk = 0.57
#> 
#> 
#> Train 1000 iterations in 0 Seconds.
#> Final risk based on the train set: 0.57
cboost
#> Component-Wise Gradient Boosting
#> 
#> Trained on PimaIndiansDiabetes with target diabetes
#> Number of base-learners: 4
#> Learning rate: 0.05
#> Iterations: 1000
#> Positive class: neg
#> Offset: 0.3118
#> 
#> LossBinomial Loss:
#> 
#>   Loss function: L(y,x) = log(1 + exp(-2yf(x))
#> 
#> 

cboost$getBaselearnerNames()
#> [1] "pressure_spline"           "age_spline"               
#> [3] "pregnant.cat_yes_category" "pregnant.cat_no_category"

selected.features = cboost$getSelectedBaselearner()
table(selected.features)
#> selected.features
#>               age_spline pregnant.cat_no_category          pressure_spline 
#>                      434                      150                      416

params = cboost$getEstimatedCoef()
str(params)
#> List of 4
#>  $ age_spline              : num [1:14, 1] 2.99 1.501 0.588 -0.535 -0.119 ...
#>  $ pregnant.cat_no_category: num [1, 1] -0.299
#>  $ pressure_spline         : num [1:24, 1] -0.8087 -0.4274 -0.0602 0.2226 0.3368 ...
#>  $ offset                  : num 0.312

cboost$train(3000)
#> 
#> You have already trained 1000 iterations.
#> Train 2000 additional iterations.

cboost$plot("age_spline", iters = c(100, 500, 1000, 2000, 3000)) +
  ggthemes::theme_tufte() + 
  ggplot2::scale_color_brewer(palette = "Spectral")
```

<p align="center">

<img src="Readme_files/cboost-1.png" width="70%" />

</p>

## Benchmark

To get an idea of the performance of compboost, we have conduct a small
benchmark in which compboost is compared with mboost. For this purpose,
the runtime behavior and memory consumption of the two packages were
compared. The results of the benchmark can be read
[here](https://github.com/schalkdaniel/compboost/tree/master/benchmark).
