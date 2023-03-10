# Release Notes

- **10.03.2023** \
  The package now supports saving without data (see the project page or vignettes for details).

- **08.03.2023** \
  A [`mlr3`](https://mlr3.mlr-org.com/) learner is now available.

- **08.03.2023** \
  Exclude `CustomCpp` base learner until further notice (due to critical errors).

- **02.03.2023** \
  Bigger refactoring of the documentation and smaller bugfixes.

- **24.02.2023** \
  Removing the `dev` branch and return to simple feature branch workflow.

- **15.01.2023** \
  Happy new year! Finally, it is possible to save and load `Compboost` objects as `JSON`.

- **21.12.2022** \
  New methods for transforming data are now available (`cboost$transformData(newdat)`). Additionally, methods for accessing the meta data of a base learner were also added (`cboost$baselearner_list$blfactory$factory$getMeta()`).

- **09.12.2022** \
  Re-writing the documentation for the `Compboost` class.

- **20.04.2022** \
  Adding a new "intercept base learner". This base learner can be used to additionally add an intercept. This makes sense if, e.g. linear functions without intercept are added.

- **20.04.2022**\
  A lot happens until the last entry. The development was done more in a rush without prober documentation. In the future, a bigger update will come containing an updated documentation and a method to store models.

- **30.04.2020** \
  Refactoring core code basis:
      - Working just with protected/private class member and corresponding setter and getter.
      - Consistent naming convention of class member.
      - Simplification of constructing classes by using a more flexible class inheritance structure.

- **24.04.2020** \
  It is now possible to choose between to solvers for fitting the base-learner. The two options are the Cholesky
  decomposition and to use the inverse.

- **15.04.2020** \
  A new base-learner `BaselearnerCategoricalBinary` is now available. This base-learner reduces the memory load and improves
  the runtime.

- **03.04.2020** \
  Binning can now be used for spline base-learner to reduce memory load and increase runtime performance. See `?compboost::BaselearnerPSpline`.

- **12.04.2019** \
  The Huber loss is now available for training.

- **08.04.2019** \
  Quantile loss for quantile regression.

- **01.03.2019** \
  It is now possible to use parallel optimizer to speed up training.

## compboost 0.1.1

- **23.01.2019** \
  Most parts of compboost are now using smart pointer.

- **23.01.2019** \
  **Style**: Change `.` to `_`, e.g. change `n.knots` to `n_knots`, to be more consistent with `C++` syntax.

- **19.01.2019** \
  There is now a new `Response` class to be more versatile for given tasks.

- **14.12.2018** \
  To track the out of bag risk is now easy controllable through a argument `oob.fraction`. The paths of inbag vs. out of bag risk can be plotted with `plotInbagVsOobRisk()`

- **28.11.2018** \
  It is now possible to directly access the logger data with `getLoggerData()` and to calculate and plot feature importance with `calculateFeatureImportance()` and `plotFeatureImportance()`.

- **27.11.2018** \
  Fix bug in the spline base-learner for out of range values.

- **09.11.2018** \
  Adding a new optimizer `OptimizerCoordinateDescentLineSearch` which conducts line search after each iteration.

- **09.11.2018** \
  Improve trace of the training process by passing logger identifier directly to `C++`.

## compboost 0.1.0

Initial release

- **19.07.2018** \
  Compboost now uses sparse matrices for splines to reduce memory load.

- **29.06.2018** \
  Compboost API is almost ready to use.

- **14.06.2018** \
  Update naming `GreedyOptimizer` -> `OptimizerCoordinateDescent` and small typos.

- **30.03.2018** \
  Compboost is now ready to do binary classification by using the
  `BernoulliLoss`.

- **29.03.2018** \
  Upload `C++` documentation created by doxygen.

- **28.03.2018** \
  P-Splines are now available as base-learner. Additionally the Polynomial and P-Spline learner
  are speed up using a more general data structure which stores the inverse once and reuse it for
  every iteration.

- **21.03.2018** \
  New data structure with independent source and target.

- **01.03.2018** \
  Compboost should now run stable and without memory leaks.

- **07.02.2018** \
  Naming of the `C++` classes. Those are matching the `R` classes now.

- **29.01.2018** \
  Update naming to a more consistent scheme.

- **26.01.2018** \
  Add printer for the classes.

- **22.01.2018** \
  Add inbag and out of bag logger.

- **21.01.2018** \
  New structure for factories and base-learner. The function
  `InstantiateData` is now member of the factory, not the base-learner. This
  should also speed up the algorithm, since we don't have to check whether data
  is instantiated or not. We can do that once within the constructor.
  Additionally, it should be more clear now what the member does since there is
  no hacky base-learner helper necessary to instantiate the data.
