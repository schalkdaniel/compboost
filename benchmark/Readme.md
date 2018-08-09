
# Benchmarking compboost vs. mboost

This benchmark was executed on a `Linux` machine with a `Intel(R)
Xeon(R) CPU E5-2650 v2 @ 2.60GHz 62GiB System` using the `R` package
`batchtools`.

This document was automatically created using `drake`. To recreate this
document just source `drake_benchmark.R`.

## Runtime Benchmark

To access the raw results you need to load the registry:

After preprocessing the raw data are stored into a `data.frame` where
each row represents a job with instances like the elapsed time and the
dimension of the simulated data:

| job.id |    time | learner | iters | algo      | nrows | ncols |
| -----: | ------: | :------ | ----: | :-------- | ----: | ----: |
|    144 |  0.8205 | spline  |   100 | gamboost  |  2000 |  1001 |
|    457 | 10.6092 | linear  |  1500 | mboost    | 10000 |  1001 |
|    255 |  1.4300 | spline  |  1500 | compboost |  2000 |  1001 |
|    492 |  0.0848 | spline  |  1500 | gamboost  |  2000 |    11 |
|    502 |  0.3844 | spline  |  1500 | gamboost  |  2000 |    51 |
|    111 |  1.5987 | linear  |   500 | mboost    |  2000 |  1001 |
|    109 |  0.3911 | linear  |   100 | mboost    |  2000 |  1001 |
|    253 |  1.4084 | spline  |  1500 | compboost |  2000 |  1001 |
|    162 | 25.2750 | spline  |  5000 | gamboost  |  2000 |  1001 |
|    225 |  0.0733 | spline  |  1500 | compboost |  2000 |    51 |

The preprocessing is defined in the `drake_runtime_benchmark.R` script
where `raw.runtime.benchmark.data` is created. This also applies for the
following graphics.

For any of the following bars with a height of zero it was not possible
to execute the algorithm with the corresponding specification.

### Increasing Number of Iterations

While increasing the number of iterations we fixed the number of
observations at 2000, and the number of feature at 1000. Under this
configuration we achieve a 15 times faster fitting process with
`compboost` compared to `mboost` in boosting linear base-learner.
Nevertheless, `glmboost` is faster due to the internal structure of
`glmboost` with that all base-learners can fitted in one matrix
multiplication. But, this approach is not suitable with `compboost`
since it does not fit into the object-oriented system we provide. This
is due to the flexibility in specifying ordinary base-learner
combination and not making the whole fitting process conditionally on
the used base-learners. Nevertheless, using spline base-learner,
`compboost` is about five times faster than `mboost` and `glmboost`
(which is just a wrapper of the original `mboost`
algorithm).

<img src="Readme_files/figure-gfm/unnamed-chunk-5-1.png" width="100%" style="display: block; margin: auto;" />

Note that the relative factor highly depends on the number of
observations. This behavior is described above.

### Increasing Number of Base-Learner

For increasing the number of base-learners we get a equal behavior as
for increasing the number of iterations. Nevertheless, with `mboost` it
was not able to conduct the boosting on 4000 features while `compboost`
it was. For this experiments we fix the number of observations at 2000
and the number of iterations at
1500.

<img src="Readme_files/figure-gfm/unnamed-chunk-7-1.png" width="100%" style="display: block; margin: auto;" />

### Increasing Number of Observations

This may have the biggest effect on computation time since increasing
the number of observations affects the allocated memory as well as the
size of the internal matrix multiplications.

<!--
- C++ meta-code is much faster then R and matrix multiplication does not have such a huge weight for small n
- With larger n, the meta-code becomes less weight and the matrix multiplications are dominating the runtime
- In some point the relative runtime has a minimum, but for larger matrix multiplications it should increase again
-->

<img src="Readme_files/figure-gfm/unnamed-chunk-8-1.png" width="100%" style="display: block; margin: auto;" />

<!--
### Memory Benchmark
-->
