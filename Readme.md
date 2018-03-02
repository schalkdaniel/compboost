[![Build Status](https://travis-ci.org/schalkdaniel/compboost.svg?branch=master)](https://travis-ci.org/schalkdaniel/compboost)
[![Coverage Status](https://coveralls.io/repos/github/schalkdaniel/compboost/badge.svg?branch=master)](https://coveralls.io/github/schalkdaniel/compboost?branch=master)

This repository is still under development. If you like you can follow the process.

## Installation

#### Developer version:
```r
devtools::install_github("schalkdaniel/compboost")
```

## Road Map

- [ ] Technical Stuff:
    - [x] Deal with destructors and remove data cleanly
    - [x] Fix compboost that it doesn't crash `R` after some time
    - [ ] Parallel computations:
        - [ ] Greedy optimizer
        - [ ] Data transformation for spline learner
    - [ ] Make the algorithm more memory friendly by using sparse matrices (if possible)
    
- [ ] Baselearner:
    - [ ] Implement P-Spline baselearner
    
- [x] Logger:
    - [x] Implement OOB Logger
    - [x] Implement inbag logger (basically done by the design of the algorithm, but it isn't tracked at the moment)
    
- [ ] General Implementation:
    - [x] Implement parameter getter
        - [x] Getter for final parameter
        - [x] Getter for parameter of iteration `k < iter.max`
        - [x] Getter for parameter matrix for all iterations
    - [ ] Prediction:
        - [x] General predict function on trian data
        - [ ] Predict function for iteration `k < iter.max`
        - [x] Prediction on newdata
        - [x] Prediction on newdata for iteration `k < iter.max`
    - [ ] Data class:
        - [ ] Abstract class setting
        - [ ] Ordinary child:
            - [ ] dataGetter
            - [ ] dataSetter
        - [ ] Out of memory child:
            - [ ] dataGetter
            - [ ] dataSetter
        
- [ ] Tests:
    - [ ] Iterate over tests (they are not coded very well)
    - [x] Test for `BaselearnerCpp` see #86
    
- [x] Naming:
    - [x] Consistent naming between `R` and `C++`

## Changelog

- **21.01.2018** \
  New structure for factorys and baselearner. The function
  `InstantiateData` is now member of the factory, not the baselearner. This 
  should also speed up the algorithm, since we don't have to check whether data
  is instantiated or not. We can do that once within the constructor. 
  Additionally, it should be more clear now what the member does since there is
  no hacky baselearner helper necessary to instantiate the data.
  
- **22.01.2018** \
  Add inbag and out of bag logger.
  
- **26.01.2018** \
  Add printer for the classes.
  
- **29.01.2018** \
  Update naming to a mroe consistent scheme.
  
- **07.02.2018** \
  Naming of the `C++` classes. Those are matching the `R` classes now.

## Idea

## Usage

## Implementation

## References
