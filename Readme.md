[![Build Status](https://travis-ci.org/schalkdaniel/compboost.svg?branch=master)](https://travis-ci.org/schalkdaniel/compboost)
[![Coverage Status](https://coveralls.io/repos/github/schalkdaniel/compboost/badge.svg?branch=master)](https://coveralls.io/github/schalkdaniel/compboost?branch=master)

This repository is still under development. If you like you can follow the process.

## Installation

#### Developer version:
```r
devtools::install_github("schalkdaniel/compboost")
```

## Road Map

- [ ] **Technical Stuff:**
    - [x] Deal with destructors and remove data cleanly
    - [x] Fix compboost that it doesn't crash `R` after some time
    - [ ] Parallel computations:
        - [ ] Greedy optimizer
        - [ ] Data transformation for spline learner
    - [ ] Make the algorithm more memory friendly by using sparse matrices (if possible)
    
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
    
- [ ] **General Implementation:**
    - [x] Implement parameter getter
        - [x] Getter for final parameter
        - [x] Getter for parameter of iteration `k < iter.max`
        - [x] Getter for parameter matrix for all iterations
    - [ ] Prediction:
        - [x] General predict function on trian data
        - [ ] Predict function for iteration `k < iter.max`
        - [x] Prediction on newdata
        - [x] Prediction on newdata for iteration `k < iter.max`
        
- [ ] **Tests:**
    - [ ] Iterate over tests (they are not coded very well)
    - [x] Test for `BaselearnerCpp` see #86
    
- [x] **Naming:**
    - [x] Consistent class naming between `R` and `C++`
    - [x] Use unified function naming

## Changelog/Updates

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


## Idea

- Everything should be as modular as possible (done by using polymorphism)

## Usage

- Fast usage via `R` API
- Highly customizable usage via classes

