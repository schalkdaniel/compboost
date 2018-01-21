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
    - [ ] Deal with destructors and remove data cleanly
    - [ ] Fix compboost that it doesn't crash `R` after some time
    
- [ ] Baselearner:
    - [ ] Implement spline baselearner
    
- [ ] Logger:
    - [ ] Implement OOB Logger
    - [ ] Implement inbag logger (basically done by the design of the algorithm, but it isn't tracked at the moment)
    
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
        
- [ ] Tests:
    - [ ] Iterate over tests (they are notd coded very well)
    - [ ] Test for `BaselearnerCpp` see #86

## Changelog

- **21.01.2018:** New structure for factorys and baselearner. The new member
  `InstantiateData` is now member of the factory, not the baselearner. This 
  should also speed up the algorithm, since we don't have to check whether data
  is instantiated or not. We can do that once within the constructor. 
  Additionally, it should be more clear now what the member does since there is
  no hacky helper baselearner necessary to instantiate the data.

## Idea

## Usage

## Implementation

## References
