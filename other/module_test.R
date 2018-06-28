library(Rcpp)
library(inline)

fx <- inline::cxxfunction(signature(), plugin="Rcpp", include=readLines("other/module_test.cpp"))

## assumes fx_unif <- cxxfunction(...) ran
unif_module <- Module("unif_module", getDynLib(fx))
Uniform <- unif_module$Uniform

Uniform@fields

u <- new(Uniform, 0, 10)
u$draw(10L)