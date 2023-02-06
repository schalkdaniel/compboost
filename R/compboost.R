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
#' @examples
#' cboost = Compboost$new(mtcars, "mpg", loss = LossQuadratic$new(), oob_fraction = 0.3)
#' cboost$addBaselearner("hp", "spline", BaselearnerPSpline, degree = 3,
#'   n_knots = 10, df = 3, differences = 2)
#' cboost$addBaselearner("wt", "spline", BaselearnerPSpline)
#' cboost$train(1000)
#'
#' table(cboost$getSelectedBaselearner())
#' head(cboost$logs)
#' names(cboost$baselearner_list)
#'
#' # Access information about the a base learner in the list:
#' cboost$baselearner_list$hp_spline$factory$getDF()
#' cboost$baselearner_list$hp_spline$factory$getPenalty()
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

    #' @field id (`character(1)`)\cr
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

    #' @field early_stop (`logical(1)`)\cr
    #' Indicator whether early stopping is used or not.
    early_stop = FALSE,

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
    #' @param early_stop (`logical(1)`)\cr
    #' Indicator whether early stopping should be used or not.
    #' @template param-idx_oob
    #' @param stop_args (`list(integer(1), integer(1))`)\cr
    #' `list` containing two elements `patience` and `eps_for_break` which are used for early stopping.
    #' @param file (`character(1`)\cr
    #' File from which a model should be loaded. If `NULL`, `data` and `target` must be defined.
    initialize = function(data = NULL, target = NULL, optimizer = NULL, loss = NULL,
      learning_rate = 0.05, positive = NULL, oob_fraction = NULL, early_stop = FALSE,
      idx_oob = NULL, stop_args = list(eps_for_break = 0, patience = 10L), file = NULL) {

      if (all(is.null(file), is.null(data), is.null(target))) {
        stop("Make sure to specify `data` and `target` or load from a file with `file = [filename].json`")
      }

      if (is.null(file)) {

        # CODE TO CREATE COMPBOOST OBJECT FROM ARGUMENTS:

        if (is.null(data)) {
          stop("Data is a required argument if no file is given.")
        } else {

          # CHECKS:
          checkmate::assertDataFrame(data, any.missing = FALSE, min.rows = 1)

        }
        if (is.null(target)) {
          stop("target is a required argument if no file is given.")
        } else {

          # CHECKS:
          if (! is.null(positive)) {
            x = data[[target]]
            ct = class(data[[target]])
            if (! inherits(ct, c("character", "factor")))
              stop("Target must be of class `character` or `factor` if `positive` is specified. Target class is: ", ct)

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
        }
        if (is.null(optimizer)) {
          optimizer = OptimizerCoordinateDescent$new()
        }
        if (is.null(loss)) {
          if (isRcppClass(target, "Response")) {
            tname = target$getTargetName()
          } else {
            tname = target
          }
          linit = FALSE
          if (is.numeric(data[[tname]])) {
            loss = LossQuadratic$new()
            linit = TRUE
          }
          if (is.character(data[[tname]]) || is.factor(data[[tname]])) {
            loss = LossBinomial$new()
            linit = TRUE
          }
          if (! linit) {
            stop("Was not able to automatically guess a loss class.")
          }
        }

        checkmate::assertNumeric(learning_rate, lower = 0, upper = 1, any.missing = FALSE, len = 1)
        checkmate::assertNumeric(oob_fraction, lower = 0, upper = 1, any.missing = FALSE, len = 1, null.ok = TRUE)
        checkmate::assertLogical(early_stop, any.missing = FALSE, len = 1L)
        checkmate::assertInteger(idx_oob, null.ok = TRUE, upper = nrow(data), unique = TRUE, any.missing = FALSE)

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
            private$p_idx_oob = sample(x = seq_len(nrow(data)), size = floor(oob_fraction * nrow(data)), replace = FALSE)
          } else {
            private$p_idx_oob = idx_oob
          }
          if ((! is.null(idx_oob)) && (! is.null(oob_fraction))) {
            warning("`oob_fraction` is ignored when a specific test index is given.")
          }
        }
        private$p_idx_train = setdiff(seq_len(nrow(data)), private$p_idx_oob)

        if (is.character(target)) {
          checkmate::assertCharacter(target)
          if (! target %in% names(data))
            stop("The target ", target, " is not present within the data")

          # With .vectorToRespone we are very restricted to the task types.
          # We can just guess for regression or classification. For every
          # other task one should use the Response interface!
          self$response = vectorToResponse(data[[target]][private$p_idx_train], target, positive)
        } else {
          assertRcppClass(target, "Response")
          if (nrow(target$getResponse()) != nrow(data))
            stop("Response must have same number of observations as the given dataset")
          self$response = target
        }

        self$oob_fraction = oob_fraction
        self$early_stop = early_stop
        self$target = self$response$getTargetName()
        self$data = data[private$p_idx_train, !colnames(data) %in% self$target, drop = FALSE]
        self$optimizer = optimizer
        self$loss = loss
        self$learning_rate = learning_rate

        if (self$early_stop || (! is.null(self$oob_fraction) || (! is.null(idx_oob)))) {
          self$data_oob = data[private$p_idx_oob, !colnames(data) %in% target, drop = FALSE]
          self$response_oob = data[private$p_idx_oob, self$target]
          self$response_oob = self$prepareResponse(self$response_oob)
        }

        # Initialize new base-learner factory list. All factories which are defined in
        # `addBaselearner` are registered here:
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
        private$p_stop_args = stop_args

      } else {

        # LOAD COMPBOOST FROM FILE:
        private$loadFromJson(file)

      }
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
    #' @param ... \cr
    #' Additional arguments passed to `loger$new(logger_id, use_as_stopper, ...)`.
    addLogger = function(logger, use_as_stopper = FALSE, logger_id, ...) {
      if (! is.null(self$model)) {
        stop("Logger can not be added after training was started")
      }
      private$p_l_list[[logger_id]] = logger$new(logger_id, use_as_stopper = use_as_stopper, ...)
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
      private$p_bl_list[[id_int]] = list()
      private$p_bl_list[[id_int]]$source = data_source$new(as.matrix(rep(1, nrow(self$data))), "intercept")
      private$p_bl_list[[id_int]]$feature = "intercept"
      private$p_bl_list[[id_int]]$factory = BaselearnerPolynomial$new(private$p_bl_list[[id_int]]$source, "",
        list(degree = 1, intercept = FALSE))

      self$bl_factory_list$registerFactory(private$p_bl_list[[id_int]]$factory)
      private$p_bl_list[[id_int]]$source = NULL
    },

    #' @description
    #' Add a base learner of one feature to the model that is considered in each iteration.
    #' Using `$addBaselearner()` just allows including univariate features. See `$addTensor()` for
    #' bivariate effect modelling and `$addComponents()` for an effect decomposition.
    #'
    #' @template param-feature
    #' @param id (`character(1)`)\cr
    #' The name of the base learner.
    #' @template param-bl_factory
    #' @template param-data_source
    #' @param ... \cr
    #' Further argument spassed to the `$new(...)` constructor of `bl_factory`.
    addBaselearner = function(feature, id, bl_factory, data_source = InMemoryData, ...) {
      if (!is.null(self$model)) {
        stop("No base-learners can be added after training is started")
      }

      # Clear base-learners which are within the bl_list but not registered:
      idx_remove = ! names(private$p_bl_list) %in% self$bl_factory_list$getRegisteredFactoryNames()
      if (any(idx_remove)) {
        for (i in which(idx_remove)) {
          private$p_bl_list[[i]] = NULL
        }
      }

      data_columns = self$data[, feature, drop = FALSE]
      id_fac = paste(paste(feature, collapse = "_"), id, sep = "_")

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
      #checkmate::assertChoice(blname, choices = names(private$p_bl_list))
      checkmate::assertChoice(blname, choices = self$bl_factory_list$getRegisteredFactoryNames())

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
      idx_remove = ! names(private$p_bl_list) %in% self$bl_factory_list$getRegisteredFactoryNames()
      if (any(idx_remove)) {
        for (i in which(idx_remove)) {
          private$p_bl_list[[i]] = NULL
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
      private$p_bl_list[[id]] = list()
      private$p_bl_list[[id]]$feature = c(feature1, feature2)
      private$p_bl_list[[id]]$factory = tensor
      self$bl_factory_list$registerFactory(private$p_bl_list[[id]]$factory)
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
      idx_remove = ! names(private$p_bl_list) %in% self$bl_factory_list$getRegisteredFactoryNames()
      if (any(idx_remove)) {
        for (i in which(idx_remove)) {
          private$p_bl_list[[i]] = NULL
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
      private$p_bl_list[[id_lin]] = list()
      private$p_bl_list[[id_lin]]$feature = feature
      private$p_bl_list[[id_lin]]$factory = fac1

      self$bl_factory_list$registerFactory(private$p_bl_list[[id_lin]]$factory)

      # Register centered spline:
      id_sp = paste0(feature, "_", feature, "_spline_centered")
      private$p_bl_list[[id_sp]] = list()
      private$p_bl_list[[id_sp]]$feature = feature
      private$p_bl_list[[id_sp]]$factory = f2cen

      self$bl_factory_list$registerFactory(private$p_bl_list[[id_sp]]$factory)
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
          if ("Rcpp_LoggerIteration" %in% vapply(private$p_l_list, class, character(1))) {
            warning("Training iterations are ignored since custom iteration logger is already defined")
          } else {
            self$addLogger(LoggerIteration, TRUE, logger_id = "_iterations", iter.max = iteration)
          }
        }
        if (self$early_stop || (! is.null(self$oob_fraction) || (! is.null(private$p_idx_oob)))) private$addOobLogger()
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
      bl_features = unique(unlist(lapply(private$p_bl_list, function(x) x$feature)))

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
    #' @param response (`vector()`)\cr
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
    #' Get design matrices of all (or a subset) base learners for a new `data.frame`.
    #'
    #' @template param-newdata
    #' @param blnames (`character()`)\cr
    #' Names of the base learners for which the design matrices are returned. If
    #' `is.null(blnames)`, compboost tries to guess all base learners that were
    #' constructed based on the feature names of `newdata`.
    #'
    #' @return
    #' `list(matrix | Matrix::Matrix)` matrices as elements.
    transformData = function(newdata, blnames = NULL) {
      if (is.null(self$model)) stop("Model must be trained first.")

      nnew = names(newdata)
      ndat = setdiff(names(self$data), self$target)
      checkmate::assertCharacter(blnames, null.ok = TRUE)
      if (! is.null(blnames)) {
        nuisance = lapply(blnames, checkmate::assertChoice, choices = names(private$p_bl_list))
      } else {
        blnames = names(private$p_bl_list)
      }
      unused_cols = nnew[! nnew %in% ndat]
      if (length(unused_cols) > 0) {
        warning(sprintf("Unused features '%s' in 'newdata' are ignored.", paste(unused_cols, collapse = ", ")))
        newdata = newdata[, setdiff(nnew, unused_cols), drop = FALSE]
        nnew = names(newdata)
      }

      nuisance = lapply(nnew, checkmate::assertChoice, choices = ndat)

      ndat = self$prepareData(newdata)
      lout = lapply(names(private$p_bl_list), function(bln) {
        bl = private$p_bl_list[[bln]]
        if (all(bl$feature %in% nnew)) {
          if (bln %in% blnames) {
            return(bl$factory$transformData(ndat)$design)
          } else {
            return(NULL)
          }
        } else {
          return(NULL)
        }
      })
      names(lout) = names(private$p_bl_list)
      lout[vapply(lout, is.null, logical(1))] = NULL
      return(lout)
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
      p = sprintf(paste("\n",
          "Component-Wise Gradient Boosting\n",
          "Target variable: %s",
          "Number of base-learners: %s",
          "Learning rate: %s",
          "Iterations: %s\n", sep = "\n"),
        self$target, self$bl_factory_list$getNumberOfRegisteredFactories(),
        self$learning_rate, self$getCurrentIteration())

      if (! is.null(self$positive))
        p = paste0(p, sprintf("\nPositive class: %s", self$positive))

      if (! is.null(self$model)){
        offset = round(self$model$getOffset(), 4)
        if (length(offset) == 1)
          p = paste0(p, sprintf("\nOffset: %s", offset))
      }
      cat(p)
      cat("\n\n")
      print(self$loss)

      return(invisible(self))
    },

    #' @description
    #' Get the estimated coefficients.
    #'
    #' @return
    #' `list(pars, offset)` with estimated coefficients/parameters and intercept/offset.
    getCoef = function() {
      bl_classes = vapply(private$p_bl_list, function(bl) class(bl$factory), character(1L))
      bl_cat = bl_classes[grepl("Categorical", bl_classes)]
      if (! is.null(self$model)) {
        pars = self$model$getEstimatedParameter()
        for (blc in intersect(names(bl_cat), names(pars))) {
          dict = private$p_bl_list[[blc]]$factory$getDictionary()
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
      bl_classes = vapply(private$p_bl_list, function(bl) class(bl$factory), character(1L))
      bl_cat = bl_classes[grepl("Categorical", bl_classes)]
      if (! is.null(self$model)) {
        pars = self$model$getEstimatedParameter()
        for (blc in intersect(names(bl_cat), names(pars))) {
          dict = private$p_bl_list[[blc]]$factory$getDictionary()
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
      return(names(private$p_bl_list))
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

      private$p_logs = as.data.frame(out_mat)
      private$p_logs$baselearner = self$getSelectedBaselearner()
      if (! "train_risk" %in% names(private$p_logs)) {
        private$p_logs = rbind(NA, private$p_logs)
        private$p_logs$train_risk = self$getInbagRisk()
        private$p_logs$baselearner[1] = "intercept"
        if ("_iterations" %in% names(private$p_logs))
          private$p_logs[["_iterations"]][1] = 0
      }
      return(private$p_logs)
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
        feats = vapply(private$p_bl_list, function(bl) paste(unique(bl$feature), collapse = "_"), character(1L))
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
  ), # end public

  active = list(

    #' @field baselearner_list (`list()`)\cr
    #' Named `list` with names `$getBaselearnerNames()`. Each elements contains
    #'
    #' * `"feature"` (`character(1)`): The name of the feature from `data`.
    #' * `"factory"` (`Baselearner*`): The raw base learner as  `factory`object. See `?Baselearner*` for details.
    baselearner_list = function(x) {
      if (! missing(x)) stop("`baselearner_list` is read only.")
      return(private$p_bl_list)
    },

    #' @field boost_intercept (`logical(1)`)\cr
    #' Logical value indicating whether an intercept base learner was added with `$addIntercept()` or not.
    boost_intercept = function(x) {
      if (! missing(x)) stop("`boost_intercept` is read only.")
      return(private$p_boost_intercept)
    },

    #' @field logs (`data.frame`)\cr
    #' Basic information such as risk, selected base learner etc. about each iteration.
    #' If `oob_data` is set, further information about the validation/oob risk is also logged.
    #' The same applies for time logging etc. Note: Using the field `logs` internally is set and updated
    #' after each call to `$getLoggerData()`. Hence, it cashes the logged data set instead of
    #' recalculating the data set as it is done for `$getLoggerData()`.
    logs = function(x) {
      if (! missing(x)) stop("`logs` is read only.")
      if (is.null(self$model)) {
        stop("Logger data can only be returned after the model has been trained.")
      }
      if (is.null(private$p_logs)) {
        private$p_logs = self$getLoggerData()
      }
      return(private$p_logs)
    },

    #' @template field-idx_oob
    idx_oob = function(x) {
      if (! missing(x)) stop("`idx_oob` is read only.")
      return(private$p_idx_oob)
    },

    #' @template field-idx_train
    idx_train = function(x) {
      if (! missing(x)) stop("`idx_train` is read only.")
      return(private$p_idx_train)
    }
  ), # end active

  private = list(
    # @field p_l_list (`list()`)\cr
    # A `list` containing the uninitialized raw logger classes, e.g. [LoggerIteration], [LoggerInbagRisk], etc.
    p_l_list = list(),

    # @field p_bl_list (`list()`)\cr
    # Named `list` with names `$getBaselearnerNames()`. Each elements contains
    #
    # * `"feature"` (`character(1)`): The name of the feature from `data`.
    # * `"factory"` (`Baselearner*`): The raw base learner as  `factory`object. See `?Baselearner*` for details.
    p_bl_list = list(),

    # @field p_logger_list ([LoggerList])\cr
    # The raw [LoggerList] object (see `?LoggerList` for details). Used to manage
    # the base learners.
    p_logger_list = list(),

    # @field p_stop_args (`list()`)\cr
    # List of the arguments used to early stop the algorithm. The possible elements are:
    # * `"eps_for_break"`: See `?LoggerOobRisk` for details.
    # * `"patience"`: Number of consecutive iterations after which is stopped if the loss didn't get better.
    # * `"loss_oob"`: Initialized loss object (see the `loss` field for details).
    p_stop_args = list(),

    # @field p_idx_oob (see field `idx_oob`).
    p_idx_oob = NULL,

    # @field p_idx_train (see field `idx_train`).
    p_idx_train = NULL,

    # @field p_boost_intercept (see field `boost_intercept).
    p_boost_intercept = FALSE,

    # @field p_logs (see field `logs`).
    p_logs = NULL,

    # @description
    # Initialize the model by calling the `$new()` constructor of the [Compboost_internal] object.
    initializeModel = function() {

      private$p_logger_list = LoggerList$new()
      lapply(private$p_l_list, function(logger) private$p_logger_list$registerLogger(logger))

      self$model = Compboost_internal$new(self$response, self$learning_rate,
        self$stop_all, self$bl_factory_list, self$loss,
        private$p_logger_list, self$optimizer)
    },

    # @description
    # Wrapper to add a logger for tracking the validation risk.
    addOobLogger = function() {
      if ("loss_oob" %in% names(private$p_stop_args)) {
        loss_oob = private$p_stop_args$loss_oob
        assertRcppClass(loss_oob, class(self$loss))
        control = "loss_oob"
      } else {
        loss_oob = eval(parse(text = paste0(gsub("Rcpp_", "", class(self$loss)), "$new()")))
        control = "new loss"
      }
      if (self$early_stop || (! is.null(self$oob_fraction)) || (! is.null(private$p_idx_oob))) {
        self$addLogger(logger = LoggerOobRisk, use_as_stopper = self$early_stop, logger_id = "oob_risk",
          used.loss = loss_oob, eps.for.break = private$p_stop_args$eps_for_break, patience = private$p_stop_args$patience,
          oob_data = self$prepareData(self$data_oob), oob.response = self$response_oob)
      }
    },

    # @description
    # Wrapper to add a numerical base learner.
    #
    # @param data_columns (`matrix()`)\cr
    # The raw data columns from `data` as matrix.
    # @param feature (`character()`)\cr
    # The feature names of the columns in `data_columns`.
    # @param id_fac (`character(1)`)\cr
    # The identifier of the base learner used to define the raw factory.
    # @param id (`character(1)`)\cr
    # The identifier of the base learner in the `list()` of base learners.
    # @template param-bl_factory
    # @template param-data_source
    # @param ... \cr
    # Additional arguments passed to the `$new(...)` call of `bl_factory`.
    addSingleNumericBl = function(data_columns, feature, id_fac, id, bl_factory, data_source, ...) {

      private$p_bl_list[[id]] = list()
      private$p_bl_list[[id]]$source = data_source$new(as.matrix(data_columns), paste(feature, collapse = "_"))
      private$p_bl_list[[id]]$feature = feature
      private$p_bl_list[[id]]$factory = bl_factory$new(private$p_bl_list[[id]]$source, id_fac, list(...))

      self$bl_factory_list$registerFactory(private$p_bl_list[[id]]$factory)
      private$p_bl_list[[id]]$source = NULL
    },

    # @description
    # Wrapper to add a categorical base learner.
    #
    # @param data_column (`data.frame()`)\cr
    # The raw data column from `data`.
    # @param feature (`character()`)\cr
    # The feature names of the columns in `data_columns`.
    # @param id_fac (`character(1)`)\cr
    # The identifier of the base learner used to define the raw factory.
    # @param id (`character(1)`)\cr
    # The identifier of the base learner in the `list()` of base learners.
    # @template param-bl_factory
    # @template param-data_source
    # @param ... \cr
    # Additional arguments passed to the `$new(...)` call of `bl_factory`.
    addSingleCatBl = function(data_column, feature, id_fac, id, bl_factory, data_source, ...) {

      private$p_bl_list[[id]] = list()
      private$p_bl_list[[id]]$source = CategoricalDataRaw$new(as.character(data_column[[feature]]), feature)
      if (bl_factory@.Data == "Rcpp_BaselearnerCategoricalRidge") {
        private$p_bl_list[[id]]$feature = feature
        private$p_bl_list[[id]]$factory = BaselearnerCategoricalRidge$new(private$p_bl_list[[id]]$source, id_fac, list(...))

        self$bl_factory_list$registerFactory(private$p_bl_list[[id]]$factory)
        private$p_bl_list[[id]]$source = NULL
      } else {
        lvls = unlist(unique(data_column))
        # Create dummy variable for each category and use that vector as data matrix. Hence,
        # if a categorical feature has 3 groups, then these 3 groups are added as 3 different
        # base-learners (unbiased feature selection).
        for (lvl in lvls) {

          cat_feat_id = paste(feature, lvl, id_fac, sep = "_")

          if (bl_factory@.Data == "Rcpp_BaselearnerCategoricalBinary") {
            private$p_bl_list[[cat_feat_id]] = list()

            #private$p_bl_list[[cat_feat_id]]$feature = paste(feature, lvl, sep = "_")
            private$p_bl_list[[cat_feat_id]]$feature = feature
            private$p_bl_list[[cat_feat_id]]$factory = bl_factory$new(
              private$p_bl_list[[id]]$source, paste0(lvl, "_", id_fac))

            self$bl_factory_list$registerFactory(private$p_bl_list[[cat_feat_id]]$factory)
            private$p_bl_list[[cat_feat_id]]$source = NULL
          } else {
            private$addSingleNumericBl(data_columns = as.matrix(as.integer(data_column == lvl)),
              feature = paste(feature, lvl, sep = "_"), id_fac = id_fac, id = cat_feat_id,
              bl_factory, data_source, ...)

            # This is important because of:
            #   1. feature in addSingleNumericBl needs to be something like cat_feature_Group1 to define the
            #      data objects correctly in a unique way.
            #   2. The feature itself should not be named with the level. Instead of that we just want the
            #      feature name of the categorical variable, such as cat_feature (important for predictions).
            private$p_bl_list[[cat_feat_id]]$feature = feature
          }
        }
      }
    },

    #' @description
    #' Load a [Compboost] object from a JSON file. Because of the underlying \code{C++} objects,
    #' it is not possible to use \code{R}'s native load and save methods.
    #'
    #' @param file (`character(1)`)\cr
    #'   A data frame containing the data.
    loadFromJson = function(file) {
      checkmate::assertFile(file, extension = c("json", "JSON", "Json"))

      self$model = Compboost_internal$new(file)
      self$learning_rate = self$model$getLearningRate()
      self$stop_all = self$model$useGlobalStopping()

      # RESPONSE:
      self$response = extractResponse(self$model$getResponse())
      self$positive = attr(self$response, "positive")
      self$target = self$response$getTargetName()

      # OPTIMIZER:
      self$optimizer = extractOptimizer(self$model$getOptimizer())

      # LOSS:
      self$loss = extractLoss(self$model$getLoss())

      # BASELEARNERLIST: TODO:
      self$bl_factory_list = self$model$getBaselearnerList()

      private$p_boost_intercept = "intercept" %in% self$bl_factory_list$getDataNames()
      private$p_bl_list = lapply(self$model$getFactoryMap(), function(f) {
        out = list(feature = f$getFeatureName(), factory = extractBaselearnerFactory(f))
      })

      # make active binding?
      dtmp = lapply(self$model$getDataMap(), extractData)
      self$data = do.call(data.frame, lapply(dtmp, function(d) {
        if (d$getDataType() == "in_memory") return(d$getData())
        if (d$getDataType() == "categorical") return(d$getRawData())
      }))

      if (FALSE) {
        # NECESSARY?
        self$response_oob = NULL # Only relevant in creating the objects?
        self$data_oob = NULL     # Only relevant in creating the objects?
        private$p_logger_list = self$model$getLoggerList()
        private$p_stop_args = list() # Only relevant when the oob logger is added.
        private$p_l_list = list() # Only relevant prior to training.
        self$early_stop = FALSE # Only relevant in constructor.

        # DROP:
        self$oob_fraction = NULL # Set to private?
        self$id = NULL # Removed
        private$p_idx_oob = NULL   # Not possible to reverse engineer but also not required?
                                   # -> Would be nice to have ... Save in Compboost object?
                                   #    Then make an active binding that points to the model$getTrainIdx method?
        private$p_idx_train = NULL # Not possible to reverse engineer but also not required? Same as above but point to the OOB logger?
      }
    }
  ) # end private
) # end Compboost
