#' Component-wise boosting
#'
#' This class wraps the `S4` class system exposed by `Rcpp` to fit a component-wise
#' boosting model. The two convenient wrapper [boostLinear()] and [boostSplines()] are
#' also creating objects of this class.
#'
#' Visualizing the internals see [plotBaselearnerTraces()], [plotBaselearner()], [plotFeatureImportance()],
#' [plotPEUni()], [plotTensor()], and [plotRisk()]. Visualizing the contribution for
#' one new observation see [plotIndividualContribution()].
#'
#' @export
#' @exapmles
#' cboost = Compboost$new(mtcars, "mpg", loss = LossQuadratic$new(), oob_fraction = 0.3)
#' cboost$addBaselearner("hp", "spline", BaselearnerPSpline, degree = 3,
#'   n_knots = 10, penalty = 2, differences = 2)
#' cboost$addBaselearner("wt", "spline", BaselearnerPSpline)
#' cboost$train(1000)
#'
#' table(cboost$getSelectedBaselearner())
Compboost = R6::R6Class("Compboost",
  public = list(

    #' @field data (`data.frame`)\cr
    #' The data used for training the model. Note: If `oob_fraction` is set, the
    #' input data is split into `data` and `data_oob`. Hence, `data` contains a
    #' subset of the input data to train the model.
    data = NULL,

    #' @field data_oob (`data.frame`)\cr
    #' An out-of-bag data set used for risk logging or early stopping. `data_oob`
    #' is split from the input data (see the `data` field).
    data_oob = NULL,

    #' @field oob_fraction (`numeric(1)`)\cr
    #' The fraction of `nrow(input data)` defining the number of observations in
    #' `data_oob`.
    oob_fraction = NULL,

    #' @field response ([ResponseRegr] | [ResponseBinaryClassif])\cr
    #' A `S4` response object. See `?ResponseRegr` or `?ResponseBinaryClassif` for help.
    #' This object holds the current prediction, pseudo residuals and functions to
    #' transform scores. Note: This response corresponds to the `data` field and holds
    #' the predictions for that `data.frame`.
    response = NULL,

    #' @field response_oob ([ResponseRegr] | [ResponseBinaryClassif])\cr
    #' A `S4` response object. See `?ResponseRegr` or `?ResponseBinaryClassif` for help.
    #' Same as `response` but for `data_oob`.
    response_oob = NULL,

    #' @field target (`character(1)`)\cr
    #' Name of the target variable in `data`.
    target = NULL,

    #' @field (`character(1)`)\cr
    #' Name of the data object defined in `$new(data, ...)`.
    id = NULL,

    #' @template field-optimizer
    optimizer = NULL,

    #' @template field-loss
    loss = NULL,

    #' @field learning_rate (`numeric(1)`)\cr
    #' The learning rate of the model. Note: Some optimizer do dynamically vary the learning rate.
    learning_rate = NULL,

    #' @field model ([Compboost_internal])\cr
    #' The internal Compboost object exported from `Rcpp`. See `?Compboost_internal` for details.
    model = NULL,

    #' @field bl_factory_list ([BlearnerFactoryList)\cr
    #' A container with all base learners. See `?BlearnerFactoryList` for details.
    bl_factory_list = NULL,

    #' @field positive (`character(1)`)\cr
    #' The positive class in the case of binary classification.
    positive = NULL,

    #' @field stop_all (`logical(1)`)\cr
    #' Indicator whether all stopper must return `TRUE` to early stop the algorithm.
    #' Comparable to `all()` if `stop_all = TRUE` and `any()` if `stop_all = FALSE`.
    stop_all = FALSE,

    #' @field use_early_stopping (`logical(1)`)\cr
    #' Indicator whether early stopping should be used or not.
    use_early_stopping = FALSE,

    #' @field logs (`data.frame`)\cr
    #' Basic information such as risk, selected base learner etc. about each iteration.
    #' If `oob_data` is set, further information about the validation/oob risk is also logged.
    #' The same applies for time logging etc.
    logs = NULL,

    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #'
    #' @param data (`data.frame`)\cr
    #' The data set to build the object. Note: This data set is completely used for training if `is.null(idx_oob)`.
    #' Otherwise, the data set is split into `data = data[idx_train, ]` and `data_oob = data[idx_oob, ]`.
    #' @param target (`character(1)`)\cr
    #' Character indicating the name of the target variable.
    #' @template param-optimizer
    #' @template param-loss
    #' @param learning_rate (`numeric(1)`)\cr
    #' Learning rate of the model (default is `0.05`).
    #' @param positive (`character(1)`)\cr
    #' The name of the positive class (in the case of binary classification).
    #' @param oob_fraction (`numeric(1)`)\cr
    #' The fraction of `nrow(input data)` defining the number of observations in
    #' `data_oob`. This argument is ignored if `idx_oob` is set.
    #' @param use_early_stopping (`logical(1)`)\cr
    #' Indicator whether early stopping should be used or not.
    #' @param idx_oob (`integer()`)\cr
    #' An index vector used to split `data` into `data = data[idx_train, ]` and `data_oob = data[idx_oob, ]`.
    #' Note: `oob_fraction` is ignored if this argument is set.
    #' @param stop_args (`list(integer(1), integer(1))`)\cr
    #' `list` containing two elements `patience` and `eps_for_break` which are used for early stopping.
    initialize = function(data, target, optimizer = OptimizerCoordinateDescent$new(), loss,
      learning_rate = 0.05, positive = NULL, oob_fraction = NULL, use_early_stopping = FALSE,
      idx_oob = NULL, stop_args = list(eps_for_break = 0, patience = 10L)) {

      checkmate::assertDataFrame(data, any.missing = FALSE, min.rows = 1)
      checkmate::assertNumeric(learning_rate, lower = 0, upper = 1, any.missing = FALSE, len = 1)
      checkmate::assertNumeric(oob_fraction, lower = 0, upper = 1, any.missing = FALSE, len = 1, null.ok = TRUE)
      checkmate::assertLogical(use_early_stopping, any.missing = FALSE, len = 1L)
      checkmate::assertInteger(idx_oob, null.ok = TRUE, upper = nrow(data), unique = TRUE, any.missing = FALSE)

      if (! is.null(positive)) {
        x = data[[target]]
        ct = class(data[[target]])
        if (! inherits(ct, c("character", "factor")))
          stop("Target must be of class `character` or `factor` if `positive` is specified. Target class is: ", cl)

        nux = length(unique(x))
        if (nux != 2)
          stop("Just binary classification is supported. The target has ", nux, " classes.")
      }
      checkmate::assertChoice(positive, unique(data[[target]]), null.ok = TRUE)

      if (! isRcppClass(target, "Response")) {
        if (! target %in% names(data)) {
          stop("The target ", target, " is not present within the data")
        }
      }
      if (inherits(loss, "C++Class")) {
        stop("Loss should be an initialized loss object by calling the constructor: ",
          deparse(substitute(loss)), "$new()")
      }

      if (! "eps_for_break" %in% names(stop_args)) stop_args[["eps_for_break"]] = 0
      if (! "patience" %in% names(stop_args)) stop_args[["patience"]] = 10L

      self$id = deparse(substitute(data))
      data = droplevels(as.data.frame(data))

      if ((! is.null(idx_oob)) || (! is.null(oob_fraction))) {
        if (is.null(idx_oob)) {
          private$idx_oob = sample(x = seq_len(nrow(data)), size = floor(oob_fraction * nrow(data)), replace = FALSE)
        } else {
          private$idx_oob = idx_oob
        }
        if ((! is.null(idx_oob)) && (! is.null(oob_fraction))) {
          warning("`oob_fraction` is ignored when a specific test index is given.")
        }
      }
      private$train_idx = setdiff(seq_len(nrow(data)), private$idx_oob)

      if (is.character(target)) {
        checkmate::assertCharacter(target)
        if (! target %in% names(data))
          stop("The target ", target, " is not present within the data")

        # With .vectorToRespone we are very restricted to the task types.
        # We can just guess for regression or classification. For every
        # other task one should use the Response interface!
        self$response = vectorToResponse(data[[target]][private$train_idx], target, positive)
      } else {
        assertRcppClass(target, "Response")
        if (nrow(target$getResponse()) != nrow(data))
          stop("Response must have same number of observations as the given dataset")
        self$response = target
      }

      self$oob_fraction = oob_fraction
      self$use_early_stopping = use_early_stopping
      self$target = self$response$getTargetName()
      self$data = data[private$train_idx, !colnames(data) %in% self$target, drop = FALSE]
      self$optimizer = optimizer
      self$loss = loss
      self$learning_rate = learning_rate

      if (self$use_early_stopping || (! is.null(self$oob_fraction) || (! is.null(idx_oob)))) {
        self$data_oob = data[private$idx_oob, !colnames(data) %in% target, drop = FALSE]
        self$response_oob = data[private$idx_oob, self$target]#, "oob_response")
        self$response_oob = self$prepareResponse(self$response_oob)
      }

      # Initialize new base-learner factory list. All factories which are defined in
      # `addBaselearners` are registered here:
      self$bl_factory_list = BlearnerFactoryList$new()

      # Check and set stop args:
      scount = 0
      if (! is.null(stop_args$oob_offset)) scount = 1
      if (length(stop_args) > scount) {
        for (nm in c("patience", "eps_for_break")) {
          if (! nm %in% names(stop_args)) stop("Cannot find ", nm, " in 'stop_args'")
        }
        checkmate::assertCount(stop_args$patience, positive = TRUE)
        checkmate::assertNumeric(stop_args$eps_for_break, len = 1L)
      }
      private$stop_args = stop_args
    },

    #' @description
    #' Add a logger to the model.
    #'
    #' @param logger ([LoggerIteration] | [LoggerTime] | [LoggerInbagRisk] | [LoggerOobRisk])\cr
    #' The uninitialized logger.
    #' @param use_as_stopper (`logical(1)`)\cr
    #' Indicator defining the logger as stopper considering it for early stopping.
    #' @param logger_id (`character(1)`)\cr
    #' The id of the logger. This allows to define two logger of the same type (`e.g. risk logging`) but with different arguments.
    #' @param ...\cr
    #' Additional arguments passed to `loger$new(logger_id, use_as_stopper, ...)`.
    addLogger = function(logger, use_as_stopper = FALSE, logger_id, ...) {
      private$l_list[[logger_id]] = logger$new(logger_id, use_as_stopper = use_as_stopper, ...)
    },

    #' @description
    #' Get the number of the current iteration.
    #'
    #' @return
    #' `integer(1)` value.
    getCurrentIteration = function() {
      if (!is.null(self$model) && self$model$isTrained()) {
        return(length(self$model$getSelectedBaselearner()))
      }	else {
        return(0)
      }
    },

    #' @description
    #' This functions adds a base learner that adjusts the intercept (if selected).
    #' Adding an intercept base learner may be necessary, e.g., when adding linear effects
    #' without intercept.
    #'
    #' @param id (`character(1)`)\cr
    #' The id of the base learner (default is `"intercept"`).
    #' @template param-data_source
    addIntercept = function(id = "intercept", data_source = InMemoryData) {
      id_int = paste0(id, "_")
      private$p_boost_intercept = TRUE
      private$bl_list[[id_int]] = list()
      private$bl_list[[id_int]]$source = data_source$new(as.matrix(rep(1, nrow(self$data))), "intercept")
      private$bl_list[[id_int]]$feature = "intercept"
      private$bl_list[[id_int]]$factory = BaselearnerPolynomial$new(private$bl_list[[id_int]]$source, "",
        list(degree = 1, intercept = FALSE))

      self$bl_factory_list$registerFactory(private$bl_list[[id_int]]$factory)
      private$bl_list[[id_int]]$source = NULL
    },

    #' @description
    #' Add a base learner of one feature to the model that is considered in each iteration.
    #' Using `$addBaselearner()` just allows including univariate features. See `$addTensor()` for
    #' bivariate effect modelling.
    #'
    #' @template param-feature
    #' @param id (`character(1)`)\cr
    #' The name of the base learner.
    #' @param bl_factory ([BaselearnerPolynomial] | [BaselearnerPSpline] | [BaselearnerCategoricalBinary] | [BaselearnerCategoricalRidge])\cr
    #' Uninitialized base learner class. See the respective help page for details.
    #' @template param-data_source
    #' @param ... \cr
    #' Further argument spassed to the `$new(...)` constructor of `bl_factory`.
    addBaselearner = function(feature, id, bl_factory, data_source = InMemoryData, ...) {
      if (!is.null(self$model)) {
        stop("No base-learners can be added after training is started")
      }

      # Clear base-learners which are within the bl_list but not registered:
      idx_remove = ! names(private$bl_list) %in% self$bl_factory_list$getRegisteredFactoryNames()
      if (any(idx_remove)) {
        for (i in which(idx_remove)) {
          private$bl_list[[i]] = NULL
        }
      }

      data_columns = self$data[, feature, drop = FALSE]
      id_fac = paste(paste(feature, collapse = "_"), id, sep = "_") #USE stringi

      if (ncol(data_columns) == 1 && !is.numeric(data_columns[, 1])) {
        private$addSingleCatBl(data_columns, feature, id, id_fac, bl_factory, data_source, ...)
      }	else {
        private$addSingleNumericBl(data_columns, feature, id, id_fac, bl_factory, data_source, ...)
      }
    },

    #' @description
    #' Remove a base learner from the model.
    #'
    #' @param blname (`character(1)`)\cr
    #' Name of the base learner that should be removed. Must be an element of `$getBaselearnerNames()`.
    rmBaselearner = function(blname) {
      checkmate::assertChoice(blname, choices = names(private$bl_list))

      self$bl_factory_list$rmBaselearnerFactory(factory_id)
    },

    #' @description
    #' Add a row-wise tensor product of features. Note: The base learner are pre-defined
    #' by the type of the feature. Numerical features uses a `BaselearnerPSpline` while categorical
    #' features are included using a `BaselearnerCategoricalRidge` base learner.
    #' To include an arbitrary tensor product requires to use the `S4` API with using
    #' `BaselearnerTensor` on two base learners of any type.
    #'
    #' @param feature1 (`character(1)`)\cr
    #' Name of the first feature. Must be an element of `names(data)`.
    #' @param feature2 (`character(1)`)\cr
    #' Name of the second feature. Must be an element of `names(data)`.
    #' @param df1 (`numeric(1)`)\cr
    #' The degrees of freedom used for the first base learner.
    #' @param df2 (`numeric(1)`)\cr
    #' The degrees of freedom used for the first base learner.
    #' @param isotrop (`logical(1)`)\cr
    #' Indicator how the two penalties should be combined, if `isotrop == TRUE`,
    #' the total degrees of freedom are uniformly distributed over the dimensions while
    #' `isotrop == FALSE` allows to define how strong each of the two dimensions is penalized.
    #' @param ... \cr
    #' Additional arguments passed to the `$new()` constructor of the [BaselearnerPSpline] class.
    addTensor = function(feature1, feature2, df1 = NULL, df2 = NULL, isotrop = FALSE, ...) {
      if (!is.null(self$model)) {
        stop("No base-learners can be added after training is started")
      }
      checkmate::assertChoice(feature1, choices = names(self$data))
      checkmate::assertChoice(feature2, choices = names(self$data))

      # Clear base-learners which are within the bl_list but not registered:
      idx_remove = ! names(private$bl_list) %in% self$bl_factory_list$getRegisteredFactoryNames()
      if (any(idx_remove)) {
        for (i in which(idx_remove)) {
          private$bl_list[[i]] = NULL
        }
      }

      args = list(...)
      if ("df" %in% names(args))
        warning("'df' were specified in '...', please use df1 and df2 to specify the degrees of freedom.")

      args1 = args2 = args
      if (! is.null(df1)) {
        args1$df = df1
        argc1 = list(df = df1)
      } else {
        argc1 = list()
      }

      if (! is.null(df2)) {
        args2$df = df2
        argc2 = list(df = df2)
      } else {
        argc2 = list()
      }

      x1 = self$data[[feature1]]
      #checkmate::assertNumeric(x1)
      if (is.numeric(x1)) {
        ds1 = InMemoryData$new(cbind(x1), feature1)
        fac1 = BaselearnerPSpline$new(ds1, "spline", args1)
      } else {
        ds1 = CategoricalDataRaw$new(x1, feature1)
        fac1 = BaselearnerCategoricalRidge$new(ds1, "categorical", argc1)
      }

      x2 = self$data[[feature2]]
      #checkmate::assertNumeric(x2)
      if (is.numeric(x2)) {
        ds2 = InMemoryData$new(cbind(x2), feature2)
        fac2 = BaselearnerPSpline$new(ds2, "spline", args2)
      } else {
        ds2 = CategoricalDataRaw$new(x2, feature2)
        fac2 = BaselearnerCategoricalRidge$new(ds2, "categorical", argc2)
      }

      tensor = BaselearnerTensor$new(fac1, fac2, "tensor", isotrop)

      # Register tensor:
      id = paste0(feature1, "_", feature2, "_tensor")
      private$bl_list[[id]] = list()
      private$bl_list[[id]]$feature = c(feature1, feature2)
      private$bl_list[[id]]$factory = tensor
      self$bl_factory_list$registerFactory(private$bl_list[[id]]$factory)
    },

    #' @description
    #' Add an effect with individual components. A linear term is added as well as
    #' a non-linear term without the linear effect. This ensures that the linear
    #' component is selected prior to the non-linear effect. The non-linear effect
    #' is only included if a deviation from a linear effect is required.
    #'
    #' Note: Internally, a [BaselearnerPolynomial] with degree one and a [BaselearnerCentered] is added.
    #' Centering a base learner makes the design matrix dense and hence memory is filled very fast.
    #' Considering binning may be an option to reduce the memory consumption.
    #'
    #' @template param-feature
    #' @param ... \cr
    #' Additional arguments passed to the `$new()` constructor of the [BaselearnerPSpline] class.
    addComponents = function(feature, ...) {
      if (!is.null(self$model)) {
        stop("No base-learners can be added after training is started")
      }

      checkmate::assertChoice(feature, choices = names(self$data))

      # Clear base-learners which are within the bl_list but not registered:
      idx_remove = ! names(private$bl_list) %in% self$bl_factory_list$getRegisteredFactoryNames()
      if (any(idx_remove)) {
        for (i in which(idx_remove)) {
          private$bl_list[[i]] = NULL
        }
      }
      x = self$data[[feature]]
      checkmate::assertNumeric(x)

      pars = list(...)
      if ("bin_root" %in% names(pars))
        broot = pars[["bin_root"]]
      else
        broot = 0

      ds1 = InMemoryData$new(cbind(x), feature)
      fac1 = BaselearnerPolynomial$new(ds1, "linear", list(degree = 1, bin_root = broot))
      fac2 = BaselearnerPSpline$new(ds1, "spline", pars)
      f2cen = BaselearnerCentered$new(fac2, fac1, "spline_centered")

      # Register linear factory:
      id_lin = paste0(feature, "_linear")
      private$bl_list[[id_lin]] = list()
      private$bl_list[[id_lin]]$feature = feature
      private$bl_list[[id_lin]]$factory = fac1

      self$bl_factory_list$registerFactory(private$bl_list[[id_lin]]$factory)

      # Register centered spline:
      id_sp = paste0(feature, "_spline_centered")
      private$bl_list[[id_sp]] = list()
      private$bl_list[[id_sp]]$feature = feature
      private$bl_list[[id_sp]]$factory = f2cen

      self$bl_factory_list$registerFactory(private$bl_list[[id_sp]]$factory)

      private$components[[feature]] = list(feature = feature, linear_id = id_lin, spline_id = id_sp, hp_spline = list(...))
    },

    #' @description
    #' This function extracts all information from components added with `$addComponents()`
    #' that defines a model. The result is an `S3` object of class
    #' `compboostExtract` that can be used for very basic operations such as
    #' predicting.
    #'
    #' Note: At the moment it is not possible to save the whole [Compboost] object
    #' and reuse it later. Using `$extractComponents()` gives the opportunity to
    #' "save" the minimal amount of data that defines the model.
    #'
    #' @return
    #' `list` with all components.
    extractComponents = function() {

      if (is.null(self$model)) {
        stop("Extraction just makes sense after compboost is trained!")
      }
      if (length(private$components) == 0) {
        stop("No registered components! Use `addComponent(feat)` instead of `addBaselearner`.")
      }

      cf = self$getCoef()

      out = lapply(private$components, function(cp) {

        if (is.null(cp$hp_spline[["degree"]])) {
          degree = 3L
        } else {
          degree = cp$hp_spline[["degree"]]
        }
        if (is.null(cp$hp_spline[["n_knots"]])) {
          n_knots = 20L
        } else {
          n_knots = cp$hp_spline[["n_knots"]]
        }
        cp[["knots"]]    = cpsp::createKnots(self$data[[cp$feature]], n_knots, degree)
        cp[["degree"]]   = degree
        cp[["rotation"]] = private$bl_list[[cp$spline_id]]$factory$getRotation()

        cf_linear = cf[[cp$linear_id]]
        if (is.null(cf_linear)) {
          cp[["coef_linear"]]  = NA
        } else {
          cp[["coef_linear"]]  = cf_linear
        }
        cf_splines = cf[[cp$spline_id]]
        if (is.null(cf_splines)) {
          cp[["coef_splines"]]  = NA
        } else {
          cp[["coef_splines"]]  = cf_splines
        }
        cp[["predict"]] = function(x) {
          checkmate::assertNumeric(x)
          if (is.na(cp$coef_linear[1])) {
            pred_lin = rep(0, length(x))
          } else {
            pred_lin = cp$coef_linear[1] + x * cp$coef_linear[2]
          }
          if (is.na(cp$coef_splines[1])) {
            pred_spline = rep(0, length(x))
          } else {
            basis       = cpsp::createSparseSplineBasis(x, cp$degree, cp$knots)
            pred_spline = as.numeric(basis %*% cp$rotation %*% cp$coef_splines)
          }
          return(list(linear = pred_lin, nonlinear = pred_spline))
        }
        return(cp)
      })
      out[["offset"]] = cf$offset
      class(out) = "compboostExtract"
      return(out)
    },

    #' @description
    #' Start fitting a model.
    #'
    #' @param iteration (`integer(1)`)\cr
    #' The maximal number of iteration. The algorithm can be stopped earlier
    #' if early stopping is active.
    #' @param trace (`integer(1)`)\cr
    #' The number of integers after which the status of the fitting is printed to the screen.
    #' The default `trace = -1` internally uses `trace = round(iteration / 40)`.
    #' To silently fit the model use `trace = 0`.
    train = function(iteration = 100, trace = -1) {

      if (self$bl_factory_list$getNumberOfRegisteredFactories() == 0) {
        stop("Could not train without any registered base-learner.")
      }

      checkmate::assertCount(iteration, positive = TRUE, null.ok = TRUE)
      checkmate::assertIntegerish(trace, lower = -1, upper = iteration, len = 1, null.ok = FALSE)

      if (trace == -1) {
        trace = round(iteration / 40)
      }

      # Check if it is necessary to add a initial iteration logger. This is not the case
      # when the user already has add one by calling `addLogger`:
      if (is.null(self$model)) {
        # If iteration is NULL, then there is no new iteration logger defined. This could be
        # used, for example, to train the algorithm an break it after a defined number of
        # hours or minutes.
        if (! is.null(iteration)) {
          # Add new logger in the case that there isn't already a custom defined one:
          if ("Rcpp_LoggerIteration" %in% vapply(private$l_list, class, character(1))) {
            warning("Training iterations are ignored since custom iteration logger is already defined")
          } else {
            self$addLogger(LoggerIteration, TRUE, logger_id = "_iterations", iter.max = iteration)
          }
        }
        if (self$use_early_stopping || (! is.null(self$oob_fraction) || (! is.null(private$idx_oob)))) private$addOobLogger()
        # After calling `initializeModel` it isn't possible to add base-learner or logger.
        private$initializeModel()
      }
      # Just call train for the initial fitting process. If the model is alredy initially trained,
      # then we use `setToIteration` to set to a lower iteration or retrain the model.
      if (! self$model$isTrained())
        self$model$train(trace)
      else {
        self$model$setToIteration(iteration, trace)
      }
      return(invisible(NULL))
    },

    #' @description
    #' Internally, each base learner is build on a [InMemoryData] object. Some
    #' methods (e.g. adding a [LoggerOobRisk]) requires to pass the data as
    #' `list(InMemoryData | CategoricalDataRaw)` with data objects as elements.
    #' This function converts the given `data.frame` into that format.
    #'
    #' @template param-newdata
    #'
    #' @return
    #' `list(InMemoryData | CategoricalDataRaw)` with data container as elements.
    #' Numeric features are wrapped by [InMemoryData] while categorical features
    #' are included with [CategoricalDataRaw].
    prepareData = function(newdata) {
      bl_features = unique(unlist(lapply(private$bl_list, function(x) x$feature)))

      if (private$p_boost_intercept)
        newdata = cbind(newdata, intercept = 1)

      nuisance = lapply(bl_features, function(blf) {
        if (! blf %in% names(newdata))
          warning("Missing feature ", blf, " in newdata. Note that this feature will be ignored in predictions.")
      })
      new_sources = list()

      blf_in_newdata = bl_features[bl_features %in% names(newdata)]
      out = lapply(blf_in_newdata, function(blf) {
        if (! is.numeric(newdata[[blf]])) {
          new_sources[[blf]] = CategoricalDataRaw$new(newdata[[blf]], blf)
        } else {
          new_sources[[blf]] = InMemoryData$new(cbind(newdata[[blf]]), blf)
        }
      })
      names(out) = blf_in_newdata
      return(out)
    },

    #' @description
    #' Same as for `$prepareData()` but for the response. Internally, `vectorToResponse()` is
    #' used to generate a [ResponseRegr] or [ResponseBinaryClassif] object.
    #'
    #' @param resonse (`vector()`)\cr
    #' A vector of type `numberic` or `categorical` that is transformed to an
    #' response object.
    #'
    #' @return
    #' [ResponseRegr] | [ResponseBinaryClassif] object.
    prepareResponse = function(response) {
      pos_class = NULL
      if (grepl(pattern = "ResponseBinaryClassif", x = class(self$response)))
        pos_class = self$response$getPositiveClass()

      return(vectorToResponse(vec = response, target = self$target, pos_class = pos_class))
    },

    #' @description
    #' Calculate predictions.
    #'
    #' @template param-newdata
    #' @param as_response (`logical(1)`)\cr
    #' In the case of binary classification, `as_response = TRUE` returns predictions as
    #' response, i.e. classes.
    #'
    #' @return
    #' Vector of predictions.
    predict = function(newdata = NULL, as_response = FALSE) {
      checkmate::assertDataFrame(newdata, null.ok = TRUE, min.rows = 1)
      if (is.null(newdata)) {
        return(self$model$getPrediction(as_response))
      } else {
        return(self$model$predict(self$prepareData(newdata), as_response))
      }
    },

    #' @description
    #' While `$predict()` returns the sum of all base learner predictions, this function
    #' returns a `list` with the predictions for each base learner.
    #'
    #' @template param-newdata
    #'
    #' @return
    #' Named `list()` with the included base learner names as names and the base learner
    #' predictions as elements.
    predictIndividual = function(newdata) {
      checkmate::assertDataFrame(newdata, null.ok = FALSE, min.rows = 1)
      return(self$model$predictIndividual(self$prepareData(newdata)))
    },

    #' @description
    #' Return the training risk of each iteration.
    #'
    #' @return
    #' `numeric()` vector of risk values or `NULL` if `$train()` was not called previously.
    getInbagRisk = function() {
      if (! is.null(self$model)) {
        # Return the risk + intercept, hence the current iteration + 1:
        return(self$model$getRiskVector()[seq_len(self$getCurrentIteration() + 1)])
      }
      return(NULL)
    },

    #' @description
    #' Get a vector with the name of the selected base learner of each iteration.
    #'
    #' @return
    #' `character()` vector of base learner names.
    getSelectedBaselearner = function() {
      if (!is.null(self$model))
        return(self$model$getSelectedBaselearner())
      return(NULL)
    },

    #' @description
    #' Printer of the object.
    #'
    #' @return
    #' Invisibly returns the object.
    print = function() {
      p = glue::glue("\n
        Component-Wise Gradient Boosting\n
        Trained on {self$id} with target {self$target}
        Number of base-learners: {self$bl_factory_list$getNumberOfRegisteredFactories()}
        Learning rate: {self$learning_rate}
        Iterations: {self$getCurrentIteration()}
        ")

      if (! is.null(self$positive))
        p = glue::glue(p, "\nPositive class: {self$positive}")

      if (! is.null(self$model)){
        offset = round(self$model$getOffset(), 4)
        if (length(offset) == 1)
          p = glue::glue(p, "\nOffset: {}")
      }
      print(p)
      print(self$loss)

      return(invisible(self))
    },

    #' @description
    #' Get the estimated coefficients.
    #'
    #' @return
    #' `list(pars, offset)` with estimated coefficients/parameters and intercept/offset.
    getCoef = function() {
      bl_classes = vapply(private$bl_list, function(bl) class(bl$factory), character(1L))
      bl_cat = bl_classes[grepl("Categorical", bl_classes)]
      if (! is.null(self$model)) {
        pars = self$model$getEstimatedParameter()
        for (blc in intersect(names(bl_cat), names(pars))) {
          dict = private$bl_list[[blc]]$factory$getDictionary()
          rownames(pars[[blc]]) = names(sort(dict))
        }
        for (i in seq_along(pars)) {
          bln = names(pars)[i]
          attr(pars[[bln]], "blclass") = unname(bl_classes[bln])
        }
        return(c(pars, offset = self$model$getOffset()))
      }
      return(NULL)
    },
    #' @description
    #' DEPRICATED use `$getCoef()` instead.
    #' Get the estimated coefficients.
    #'
    #' @return
    #' `list(pars, offset)` with estimated coefficients/parameters and intercept/offset.
    getEstimatedCoef = function() {
      message("Depricated, use `$getCoef()` instead.")
      bl_classes = vapply(private$bl_list, function(bl) class(bl$factory), character(1L))
      bl_cat = bl_classes[grepl("Categorical", bl_classes)]
      if (! is.null(self$model)) {
        pars = self$model$getEstimatedParameter()
        for (blc in intersect(names(bl_cat), names(pars))) {
          dict = private$bl_list[[blc]]$factory$getDictionary()
          rownames(pars[[blc]]) = names(sort(dict))
        }
        for (i in seq_along(pars)) {
          bln = names(pars)[i]
          attr(pars[[bln]], "blclass") = unname(bl_classes[bln])
        }
        return(c(pars, offset = self$model$getOffset()))
      }
      return(NULL)
    },

    #' @description
    #' Get the names of the registered base learners.
    #'
    #' @return
    #' `charcter()` of base learner names.
    getBaselearnerNames = function() {
      return(names(private$bl_list))
    },

    #' @description
    #' Get the logged information.
    #'
    #' @return
    #' `data.frame` of logging information.
    getLoggerData = function() {
      checkModelPlotAvailability(self, check_ggplot = FALSE)

      out_list = self$model$getLoggerData()
      out_mat = out_list[[2]]
      colnames(out_mat) = out_list[[1]]

      self$logs = as.data.frame(out_mat)
      self$logs$baselearner = self$getSelectedBaselearner()
      if (! "train_risk" %in% names(self$logs)) {
        self$logs = rbind(NA, self$logs)
        self$logs$train_risk = self$getInbagRisk()
        self$logs$baselearner[1] = "intercept"
        if ("_iterations" %in% names(self$logs))
          self$logs[["_iterations"]][1] = 0
      }
      return(self$logs)
    },

    #' @description
    #' Calculate feature important based on the training risk. Note that early
    #' stopping should be used to get adequate importance measures.
    #'
    #' @param num_feats (`integer(1)`)\cr
    #' The number considered features, the `num_feats` most important feature names and
    #' the respective value is returned.
    #' @param aggregate_bl_by_feat (`logical(1)`)\cr
    #' Indicator whether the importance is aggregated based on feature level. For example,
    #' adding components included two different base learners for the same feature. If
    #' `aggregate_bl_by_feat == TRUE`, the importance of these two base learners is aggregated
    #' instead of considering them individually.
    #'
    #' @return
    #' Named `numeric()` vector of length `num_feats` (if at least `num_feats` were selected)
    #' with importance values as elements.
    calculateFeatureImportance = function(num_feats = NULL, aggregate_bl_by_feat = FALSE) {
      checkModelPlotAvailability(self, check_ggplot = FALSE)

      inbag_risk_differences = abs(diff(self$getInbagRisk()))
      selected_learner = self$getSelectedBaselearner()
      fcol = "baselearner"
      if (aggregate_bl_by_feat) {
        feats = vapply(private$bl_list, function(bl) bl$feature, character(1L))
        selected_learner = feats[selected_learner]
        fcol = "feature"
      }

      max_feats = length(unique(selected_learner))
      checkmate::assert_integerish(x = num_feats, lower = 1, upper = max_feats,
        any.missing = FALSE, len = 1L, null.ok = TRUE)

      if (is.null(num_feats)) {
        num_feats = max_feats
        if (num_feats > 15L) num_feats = 15L
      }

      blearner_sums = aggregate(inbag_risk_differences, by = list(selected_learner), FUN = sum)
      colnames(blearner_sums) = c(fcol, "risk_reduction")

      out = blearner_sums[order(blearner_sums[["risk_reduction"]], decreasing = TRUE)[seq_len(num_feats)], ]
      return(out)
    }
  ),
  active = list(
    ##### CONTINUE HERE WITH DOCUMENTATION
    baselearner_list = function(x) {
      if (! missing(x)) stop("`baselearner_list` is read only.")
      return(private$bl_list)
    },
    boost_intercept = function(x) {
      if (! missing(x)) stop("`boost_intercept` is read only.")
      return(private$p_boost_intercept)
    }
  ),
  private = list(
    # Lists of single logger and base-learner factories. Necessary to prevent the factories from the
    # garbage collector which deallocates all the data from the heap and couses R to crash.
    l_list = list(),
    bl_list = list(),
    logger_list = list(),
    stop_args = list(),
    idx_oob = NULL,
    train_idx = NULL,
    components = list(),
    p_boost_intercept = FALSE,

    initializeModel = function() {

      private$logger_list = LoggerList$new()
      lapply(private$l_list, function(logger) private$logger_list$registerLogger(logger))

      self$model = Compboost_internal$new(self$response, self$learning_rate,
        self$stop_all, self$bl_factory_list, self$loss,
        private$logger_list, self$optimizer)
    },
    addOobLogger = function() {
      if ("loss_oob" %in% names(private$stop_args)) {
        loss_oob = private$stop_args$loss_oob
        assertRcppClass(loss_oob, class(self$loss))
        control = "loss_oob"
      } else {
        loss_oob = eval(parse(text = paste0(gsub("Rcpp_", "", class(self$loss)), "$new()")))
        control = "new loss"
      }
      if (self$use_early_stopping || (! is.null(self$oob_fraction)) || (! is.null(private$idx_oob))) {
        self$addLogger(logger = LoggerOobRisk, use_as_stopper = self$use_early_stopping, logger_id = "oob_risk",
          used.loss = loss_oob, eps.for.break = private$stop_args$eps_for_break, patience = private$stop_args$patience,
          oob_data = self$prepareData(self$data_oob), oob.response = self$response_oob)
      }
    },
    addSingleNumericBl = function(data_columns, feature, id_fac, id, bl_factory, data_source, ...) {

      private$bl_list[[id]] = list()
      private$bl_list[[id]]$source = data_source$new(as.matrix(data_columns), paste(feature, collapse = "_"))
      private$bl_list[[id]]$feature = feature
      private$bl_list[[id]]$factory = bl_factory$new(private$bl_list[[id]]$source, id_fac, list(...))

      self$bl_factory_list$registerFactory(private$bl_list[[id]]$factory)
      private$bl_list[[id]]$source = NULL
    },
    addSingleCatBl = function(data_column, feature, id_fac, id, bl_factory, data_source, ...) {

      private$bl_list[[id]] = list()
      private$bl_list[[id]]$source = CategoricalDataRaw$new(as.character(data_column[[feature]]), feature)
      if (bl_factory@.Data == "Rcpp_BaselearnerCategoricalRidge") {
        private$bl_list[[id]]$feature = feature
        private$bl_list[[id]]$factory = BaselearnerCategoricalRidge$new(private$bl_list[[id]]$source, id_fac, list(...))

        self$bl_factory_list$registerFactory(private$bl_list[[id]]$factory)
        private$bl_list[[id]]$source = NULL
      } else {
        lvls = unlist(unique(data_column))
        # Create dummy variable for each category and use that vector as data matrix. Hence,
        # if a categorical feature has 3 groups, then these 3 groups are added as 3 different
        # base-learners (unbiased feature selection).
        for (lvl in lvls) {

          cat_feat_id = paste(feature, lvl, id_fac, sep = "_")

          if (bl_factory@.Data == "Rcpp_BaselearnerCategoricalBinary") {
            private$bl_list[[cat_feat_id]] = list()

            #private$bl_list[[cat_feat_id]]$feature = paste(feature, lvl, sep = "_")
            private$bl_list[[cat_feat_id]]$feature = feature
            private$bl_list[[cat_feat_id]]$factory = bl_factory$new(
              private$bl_list[[id]]$source, paste0(lvl, "_", id_fac))

            self$bl_factory_list$registerFactory(private$bl_list[[cat_feat_id]]$factory)
            private$bl_list[[cat_feat_id]]$source = NULL
          } else {
            private$addSingleNumericBl(data_columns = as.matrix(as.integer(data_column == lvl)),
              feature = paste(feature, lvl, sep = "_"), id_fac = id_fac, id = cat_feat_id,
              bl_factory, data_source, ...)

            # This is important because of:
            #   1. feature in addSingleNumericBl needs to be something like cat_feature_Group1 to define the
            #      data objects correctly in a unique way.
            #   2. The feature itself should not be named with the level. Instead of that we just want the
            #      feature name of the categorical variable, such as cat_feature (important for predictions).
            private$bl_list[[cat_feat_id]]$feature = feature
          }
        }
      }
    }
  )
)

  #x1 = runif(1000L)
  #x2 = runif(1000L)
  #x3 = runif(1000L)
  #x4 = runif(1000L)

  #f1  = function(x1, x2) (1 - x1) * x2^2 + x1 * sin(pi * x2)
  #f2 = function(x1, x2) x1^2 + x2^2
  #y  = f1(x1, x2) + f2(x3, x4) + rnorm(1000L, 0, 0.1)

  #df = data.frame(x1, x2, x3, x4, y)

  #cboost = Compboost$new(data = df, target = "y", loss = LossQuadratic$new(), learning_rate = 0.1)
  #cboost = Compboost$new(data = df, target = "y", loss = LossQuadratic$new(), learning_rate = 0.1, oob_fraction = 0.33)

  #cboost$addBaselearner("x1", "spline", BaselearnerPSpline)

  #cboost$addTensor("x1", "x2", n_knots = 10, df1 = 10, df2 = 10)
  #cboost$addTensor("x3", "x4", n_knots = 10, df1 = 10, df2 = 10)

  #cboost$addLogger(LoggerOobRisk, use_as_stopper = TRUE, logger_id = "oob_risk",
    #used_loss = LossQuadratic$new(), eps_for_break = 0, patience = 5L, oob_data = cboost$prepareData(df[idx_oob, ]),
    #oob_response = cboost$prepareResponse(df$y[idx_oob]))

  #cboost$train(500)


