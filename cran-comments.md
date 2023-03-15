## Test environments
* local arch linux (kernel 6.1.4-arch1-1), R 4.2.2

Remote environments via GitHub Actions, checks are run with `--as-cran`
* macOS-latest (release), R 4.2.2
* windows-latest (release), R 4.2.2
* ubuntu-latest (devel), R Under development (unstable) (2023-03-13 r83977)
* ubuntu-latest (release), R 4.2.3
* ubuntu-latest (oldrel-1), R 4.1.3
* ubuntu-latest (oldrel-2), R 4.0.5
* win-builder (devel) with `devtools::check_win_devel()`

## R CMD check results
There were no ERRORs or WARNINGs.

There was 2 NOTE:

```
checking installed package size ... NOTE
installed size is  7.5Mb
sub-directories of 1Mb or more:
  doc    1.9Mb
  libs   4.6Mb
```

* This is a new release.
* Shared object created by Rcpp is greater than 1Mb.
* Using math equations in the vignettes increases the size of two vignettes to ~800 kB and ~600 kB.

```
checking HTML version of manual ... [10s] NOTE
Found the following HTML validation problems:
mlr_learners.compboost.html:63:6 (mlr_learners.compboost.Rd:27): Warning: missing </span> before <p>
mlr_learners.compboost.html:63:92 (mlr_learners.compboost.Rd:27): Warning: inserting implicit <span>
mlr_learners.compboost.html:66:6 (mlr_learners.compboost.Rd:28): Warning: missing </span> before <p>
...
```

* I was not able to reproduce the error on my machine. Hence it is tough to debug the problem here.
