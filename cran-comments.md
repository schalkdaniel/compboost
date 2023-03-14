## Test environments
* local arch linux (kernel 6.1.4-arch1-1), R 4.2.2

Remote environments via GitHub Actions, checks are run with `--as-cran`
* macOS-latest (release), R 4.2.2
* windows-latest (release), R 4.2.2
* ubuntu-latest (devel), R Under development (unstable) (2023-03-12 r83975)
* ubuntu-latest (release), R 4.2.2
* ubuntu-latest (oldrel-1), R 4.1.3
* ubuntu-latest (oldrel-2), R 4.0.5

## R CMD check results
There were no ERRORs or WARNINGs.

There was 1 NOTE:

* checking installed package size ... NOTE
  installed size is  7.5Mb
  sub-directories of 1Mb or more:
    doc    1.9Mb
    libs   4.6Mb
  Shared object created by Rcpp is greater than 1Mb.
