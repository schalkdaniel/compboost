## Test environments
* local arch linux (kernel 6.1.4-arch1-1), R 4.2.2

Remote environments via GitHub Actions, checks are run with `--as-cran`
* macOS-latest (release), R 4.2.3
* windows-latest (release), R 4.2.3
* ubuntu-latest (devel), R Under development (unstable) (2023-03-27 r84084)
* ubuntu-latest (release), R 4.2.3
* ubuntu-latest (oldrel-1), R 4.1.3
* ubuntu-latest (oldrel-2), R 4.0.5
* win-builder (devel) with `devtools::check_win_devel()`

## R CMD check results
There were no ERRORs or WARNINGs.

There were 2 NOTEs:

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

## CRAN submission notes
* Note from CRAN:
  | Please always add all authors, contributors and copyright holders in the
  | Authors@R field with the appropriate roles.
  |  From CRAN policies you agreed to:
  | "The ownership of copyright and intellectual property rights of all
  | components of the package must be clear and unambiguous (including from
  | the authors specification in the DESCRIPTION file). Where code is copied
  | (or derived) from the work of others (including from R itself), care
  | must be taken that any copyright/license statements are preserved and
  | authorship is not misrepresented.
  | Preferably, an ‘Authors@R’ would be used with ‘ctb’ roles for the
  | authors of such code. Alternatively, the ‘Author’ field should list
  | these authors as contributors. Where copyrights are held by an entity
  | other than the package authors, this should preferably be indicated via
  | ‘cph’ roles in the ‘Authors@R’ field, or using a ‘Copyright’ field (if
  | necessary referring to an inst/COPYRIGHTS file)." e.g.: Copyrightholders
  | of 'date.h'
  | Please explain in the submission comments what you did about this issue.
* Copyright holders of `date.h` and `json.hpp` were added within a `inst/COPYRIGHTS` file that is referenced in the `Copyright` field of the `DESCRIPTION` file.
