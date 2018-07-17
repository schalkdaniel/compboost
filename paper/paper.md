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



![Example figure.](cboost_viz.png)

<!-- Mentions (if applicable) of any ongoing research projects using the software or recent scholarly publications enabled by it -->

- Reference to mboost and why we write compboost (faster, better modular principle, ...)


# Acknowledgements

<!-- A list of key references including a link to the software archive -->
# References