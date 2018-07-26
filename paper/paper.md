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

In high-dimensional prediction problems, especially if the number of features greatly exceeds the number of observations, feature selection in an essential tool. One fundamental method for these problems is component-wise gradient boosting, which applies the boosting framework to additive statistical models, e.g., general additive models using component-wise smoothing splines [@schmid2008boosting]. Boosting these kinds of models maintains interpretability and enables unbiased model selection in high-dimensional feature spaces [@hofner2011framework].

The `R` [@R] package `compboost` is an implementation of component-wise boosting written in `C++` using`Armadillo` [@sanderson2016armadillo] to achieve fast runtime and full memory control. The central motivation is to provide a modular object oriented system which can be extended with new base-learners, loss functions, optimization strategies, and stopping criteria at runtime.  These extensions can be written in `R` for easy prototyping or directly in `C++` for optimized speed.This allows researchers to easily implement more specialized base-learners, e.g., for spatial or random effects, used in their respective research area.

Visualization of selected effects, adjusting the algorithm by efficiently setting it to different iterations efficiently jumping back and forth between iterations of the algorithm, and traces of selected base-learners and losses to obtain information about the feature importance are supported.

Compared to the reference implementation for component-wise gradient boosting in `R`, `mboost` [@mboost1], `compboost` is optimized for larger datasets and easier to extend, even though it currently lacks some of the large functionality `mboost` provides. 

The modular design of `compboost` allows extension to more complicated settings like functional data or survival analysis. Further research for component-wise gradient boosting on parallel computation, better feature selection, faster optimization techniques such as momentum and adaptive learning rates, as well as reduced overfitting areis conducted based on `compboost`. 

<!-- A list of key references including a link to the software archive -->
# References