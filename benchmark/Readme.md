
# Benchmarking compboost vs. mboost

This benchmark was executed on a `Linux` machine with a `Intel(R)
Xeon(R) CPU E5-2650 v2 @ 2.60GHz 62GiB System` using the `R` package
`batchtools`.

This document was automatically created using `drake`. To recreate this
document just source `drake_runtime_benchmark.R`.

## Runtime Benchmark

To access the raw results of the runtime benchmark you need to load the
registry:

After preprocessing the raw data are stored into a `data.frame` where
each row represents a job with instances like the elapsed time and the
dimension of the simulated data:

|     | job.id |    time | learner | iters | algo      | nrows | ncols |
| --- | -----: | ------: | :------ | ----: | :-------- | ----: | ----: |
| 144 |    144 |  0.8205 | spline  |   100 | gamboost  |  2000 |  1001 |
| 457 |    457 | 10.6092 | linear  |  1500 | mboost    | 10000 |  1001 |
| 255 |    255 |  1.4300 | spline  |  1500 | compboost |  2000 |  1001 |
| 492 |    492 |  0.0848 | spline  |  1500 | gamboost  |  2000 |    11 |
| 502 |    502 |  0.3844 | spline  |  1500 | gamboost  |  2000 |    51 |
| 111 |    111 |  1.5987 | linear  |   500 | mboost    |  2000 |  1001 |
| 109 |    109 |  0.3911 | linear  |   100 | mboost    |  2000 |  1001 |
| 253 |    253 |  1.4084 | spline  |  1500 | compboost |  2000 |  1001 |
| 162 |    162 | 25.2750 | spline  |  5000 | gamboost  |  2000 |  1001 |
| 225 |    225 |  0.0733 | spline  |  1500 | compboost |  2000 |    51 |

The preprocessing can be reproduced by taking a look at how
`raw.runtime.benchmark.data` was created within the
`drake_runtime_benchmark.R` script. This also applies for the following
graphics.

### Increasing Number of Iterations

<img src="Readme_files/figure-gfm/unnamed-chunk-4-1.png" width="100%" style="display: block; margin: auto;" />

### Increasing Number of Base-Learner

<img src="Readme_files/figure-gfm/unnamed-chunk-5-1.png" width="100%" style="display: block; margin: auto;" />

### Increasing Number of Observations

<img src="Readme_files/figure-gfm/unnamed-chunk-6-1.png" width="100%" style="display: block; margin: auto;" />

<!--
### Memory Benchmark
-->
