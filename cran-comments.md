## Test environments
* local ubuntu 18.04, R 3.5.1
* ubuntu 14.04 (on travis-ci), R 3.5.0
* local Windows 10, R 3.5.1
* Windows Server 2012 (on Appveyor), R 3.5.1
* win-builder

## R CMD check results
There were no ERRORs or WARNINGs. 

There was 1 NOTE:

* checking installed package size ... NOTE
  installed size is 24.7Mb
  sub-directories of 1Mb or more:
    libs  21.5Mb

  Shared object created by Rcpp is greater than 1Mb.


## Downstream dependencies
I have also run R CMD check on downstream dependencies of compboost using `devtools::revdep_check()`
(https://github.com/schalkdaniel/compboost/revdep). 
All packages that I could install passed.
