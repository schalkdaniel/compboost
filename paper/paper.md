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

In high-dimensional prediction problems, especially in the $p \geq n$ situation, feature selection is an essential tool. A fundamental method for problem of this type is component-wise gradient boosting, which automatically selects from a pool of base learners -- e.g. simple linear effects or component-wise smoothing splines [@schmid2008boosting] --  and produces a sparse additive statistical model. Boosting these kinds of models maintains interpretability and enables unbiased model selection in high-dimensional feature spaces [@hofner2011framework].

The `R` [@R] package `compboost`, which is actively developed on GitHub (https://github.com/schalkdaniel/compboost), implements component-wise boosting in `C++` using `Rcpp` [@eddelbuettel2013seamless] and `Armadillo` [@sanderson2016armadillo] to achieve efficient runtime behavior and full memory control. It provides a modular object-oriented system which can be extended with new base-learners, loss functions, optimization strategies, and stopping criteria, either in `R` for convenient prototyping or directly in `C++` for optimized speed. The latter extensions can be added at runtime, without recompiling the whole framework. This allows researchers to easily implement more specialized base-learners, e.g., for spatial or random effects, used in their respective research area.

Visualization of selected effects, efficient adjustment of the number of iterations and traces of selected base-learners and losses to obtain information about feature importance are supported.

Compared to the reference implementation for component-wise gradient boosting in `R`, `mboost` [@mboost1], `compboost` is optimized for larger datasets and easier to extend, even though it currently lacks some of the large functionality `mboost` provides. A detailed benchmark against `mboost` can be viewed on the [project homepage](https://compboost.org) and on [GitHub](https://github.com/schalkdaniel/compboost/tree/master/benchmark).

The modular design of `compboost` allows extension to more complicated settings like functional data or survival analysis. Further work on the package should include parallelized boosting, better feature selection, faster optimization techniques such as momentum and adaptive learning rates, as well as better overfitting control.


<!-- A list of key references including a link to the software archive -->
# References