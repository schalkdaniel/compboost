---
title: 'compboost: Modular Framework for Component-Wise Boosting'
authors:
- affiliation: 1
  name: Daniel Schalk
  orcid: 0000-0003-0950-1947
- affiliation: 1
  name: Janek Thomas
  orcid: 0000-0003-4511-6245
- affiliation: 1
  name: Bernd Bischl
  orcid: 0000-0001-6002-6980
date: "04 July 2018"
output: pdf_document
bibliography: paper.bib
tags:
- R
- machine learning
- boosting
affiliations:
- index: 1
  name: Department of Statistics, LMU Munich
---

# Summary
<!-- A clear statement of need that illustrates the purpose of the software-->

Component-wise boosting applies the boosting framework to statistical models, e.~g., general additive models using component-wise smoothing splines [@schmid2008boosting]. Boosting these kinds of models maintains interpretability and enables unbiased model selection in high dimensional feature spaces.

The `R` [@R] package `compboost` is an implementation of component-wise boosting written in `C++` to obtain high runtime performance and full memory control. The main idea is to provide a modular class system which can be extended without editing the source code. Therefore, it is possible to use R functions as well as C++ functions for custom base-learners, losses, logging mechanisms or stopping criteria. 

<!-- A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience-->

The pacakge provides two high level wrapper functions `boostLinear()` and `boostSplines()` to boost linear models or general additive models using p-splines of each numerical feature:
```r
library(compboost)

# Load data set with binary classification task:
data(PimaIndiansDiabetes, package = "mlbench")

# Quadratic loss as ordinary regression loss:
mod = boostSplines(data = PimaIndiansDiabetes, target = "diabetes", 
	loss = BinomialLoss$new())
```

The resulting model is an object using `R6`. Hence, `mod` has member functions to access the elements of the model such as the names of registered base-learner, selected base-learner, the estimated parameter or continue the training:
```r
mod$getBaselearnerNames()
## [1] "pregnant_spline" "glucose_spline"  "pressure_spline" "triceps_spline" 
## [5] "insulin_spline"  "mass_spline"     "pedigree_spline" "age_spline" 


selected.features = mod$selected()
table(selected.features)
## selected.features
##    age_spline glucose_spline    mass_spline 
##            23             61             16 

params = mod$coef()
str(params)
## List of 4
##  $ age_spline    : num [1:24, 1] 0.40127 0.25655 0.14807 0.11766 -0.00586 ...
##  $ glucose_spline: num [1:24, 1] -0.2041 0.0343 0.2703 0.4921 0.6856 ...
##  $ mass_spline   : num [1:24, 1] 0.0681 0.0949 0.1216 0.1473 0.1714 ...
##  $ offset        : num 0.312

mod$train(3000)
## 
## You have already trained 100 iterations.
## Train 2900 additional iterations.
## 
```

Additionally it is possible to visualize the effect of single features by calling the member function `plot()` of a specific learner. Additionally, it is possible to pass a vector of iterations used for the graphic:
```r
mod$plot("age_spline", iters = c(100, 500, 1000, 2000, 3000))
```
![Visualize compboost](cboost_viz.png)

<!-- Mentions (if applicable) of any ongoing research projects using the software or recent scholarly publications enabled by it -->

- Reference to mboost and why we write compboost (faster, better modular principle, ...)


# Acknowledgements

<!-- A list of key references including a link to the software archive -->
# References