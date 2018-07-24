[![Build Status](https://travis-ci.org/schalkdaniel/compboost.svg?branch=master)](https://travis-ci.org/schalkdaniel/compboost)
[![Coverage Status](https://coveralls.io/repos/github/schalkdaniel/compboost/badge.svg?branch=master)](https://coveralls.io/github/schalkdaniel/compboost?branch=master)

## compboost: Fast and Flexible Component-Wise Boosting Framework

Component-wise boosting applies the boosting framework to
statistical models, e.g., general additive models using component-wise smoothing 
splines. Boosting these kinds of models maintains interpretability and enables 
unbiased model selection in high dimensional feature spaces. 

The `R` package `compboost` is an alternative implementation of component-wise
boosting written in `C++` to obtain high runtime
performance and full memory control. The main idea is to provide a modular
class system which can be extended without editing the
source code. Therefore, it is possible to use `R` functions as well as
`C++` functions for custom base-learners, losses, logging mechanisms or 
stopping criteria. 

For an introduction and overview about the functionality visit the [project page](https://schalkdaniel.github.io/compboost/).

## Installation

#### Developer version:

```r
devtools::install_github("schalkdaniel/compboost")
```

## Examples

```r
library(compboost)

# Load data set with binary classification task:
data(PimaIndiansDiabetes, package = "mlbench")

# Use Binomial loss for binary classification:
cboost = boostSplines(data = PimaIndiansDiabetes, target = "diabetes", 
  loss = BinomialLoss$new())

cboost$getBaselearnerNames()
## [1] "pregnant_spline" "glucose_spline"  "pressure_spline" "triceps_spline" 
## [5] "insulin_spline"  "mass_spline"     "pedigree_spline" "age_spline" 


selected.features = cboost$selected()
table(selected.features)
## selected.features
##    age_spline glucose_spline    mass_spline 
##            23             61             16 

params = cboost$coef()
str(params)
## List of 4
##  $ age_spline    : num [1:24, 1] 0.40127 0.25655 0.14807 0.11766 -0.00586 ...
##  $ glucose_spline: num [1:24, 1] -0.2041 0.0343 0.2703 0.4921 0.6856 ...
##  $ mass_spline   : num [1:24, 1] 0.0681 0.0949 0.1216 0.1473 0.1714 ...
##  $ offset        : num 0.312

cboost$train(3000)
## 
## You have already trained 100 iterations.
## Train 2900 additional iterations.
## 

cboost$plot("age_spline", iters = c(100, 500, 1000, 2000, 3000))
```
<p align="center">
  <img src="docs/images/cboost_viz.png" alt="Compboost Visualization" width="70%">
</p>

<!--
## Road Map

- [ ] **Technical Stuff:**
    - [x] Deal with destructors and remove data cleanly
    - [x] Fix compboost that it doesn't crash `R` after some time
    - [ ] Parallel computations:
        - [ ] Greedy optimizer
        - [ ] Data transformation for spline learner
    - [x] Make the algorithm more memory friendly by using sparse matrices (if possible)
    
- [x] **Classes:**    
    - [x] Baselearner Class:
        - [x] Implement B-Spline baselearner
        - [x] Implement P-Spline baselearner
    
    - [x] Logger Class:
        - [x] Implement OOB Logger
        - [x] Implement inbag logger (basically done by the design of the algorithm, but it isn't tracked at the moment)
    
    - [ ] Data Class:
        - [x] Abstract class setting
        - [x] In memory child:
            - [x] dataGetter
            - [x] dataSetter
        - [ ] Out of memory child:
            - [ ] dataGetter
            - [ ] dataSetter
    
- [x] **General Implementation:**
    - [x] Implement parameter getter
        - [x] Getter for final parameter
        - [x] Getter for parameter of iteration `k < iter.max`
        - [x] Getter for parameter matrix for all iterations
    - [x] Prediction:
        - [x] General predict function on trian data
        - [x] Predict function for iteration `k < iter.max`
        - [x] Prediction on newdata
        - [x] Prediction on newdata for iteration `k < iter.max`
        
- [x] **Tests:**
    - [x] Iterate over tests (they are not coded very well)
    - [x] Test for `BaselearnerCpp` see #86
    
- [x] **Naming:**
    - [x] Consistent class naming between `R` and `C++`
    - [x] Use unified function naming

-->

## License

© 2018 [Daniel Schalk](https://danielschalk.com)

The contents of this repository are distributed under the MIT license. See below for details:

    The MIT License (MIT)

    Copyright (c) 2018 Daniel Schalk

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

## Changelog/Updates

- **19.07.2018** \
  Compboost now uses sparse matrices for splines to reduce memory load.

- **29.06.2018** \
  Compboost API is almost ready to use.
  
- **14.06.2018** \
  Update naming `GreedyOptimizer` -> `CoordinateDescent` and small typos.

- **30.03.2018** \
  Compboost is now ready to do binary classification by using the 
  `BernoulliLoss`.
  
- **29.03.2018** \
  Upload `C++` documentation created by doxygen. 

- **28.03.2018** \
  P-Splines are now availalbe as baselearner. Additionally the Polynomial and P-Spline learner
  are speeded up using a more gneeral data structure which stores the inverse once and reuse it for
  every iteration.

- **21.03.2018** \
  New data structure with independent source and target.
  
- **01.03.2018** \
  Compboost should now run stable and without memory leaks.

- **07.02.2018** \
  Naming of the `C++` classes. Those are matching the `R` classes now.

- **29.01.2018** \
  Update naming to a mroe consistent scheme.
  
- **26.01.2018** \
  Add printer for the classes.
  
- **22.01.2018** \
  Add inbag and out of bag logger.
  
- **21.01.2018** \
  New structure for factorys and baselearner. The function
  `InstantiateData` is now member of the factory, not the baselearner. This 
  should also speed up the algorithm, since we don't have to check whether data
  is instantiated or not. We can do that once within the constructor. 
  Additionally, it should be more clear now what the member does since there is
  no hacky baselearner helper necessary to instantiate the data.
