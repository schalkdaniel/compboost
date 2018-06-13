Compboost = R6::R6Class("Compboost",
  public = list(
    data = NULL,
    response = NULL,
    optimizer = NULL,
    loss = NULL,
    learning.rate = NULL,
    model = NULL,
    is.initialized = FALSE,
    bl.factory.list = NULL,
    logger.list = list(),
    stop.if.all.stoppers.fulfilled = FALSE,
    initialize = function(data, target, optimizer = GreedyOptimizer$new(), loss, learning.rate = 0.05) {
      checkmate::assertDataFrame(data, any.missing = FALSE, min.rows = 1)
      checkmate::assertCharacter(target)
      checkmate::assertNumeric(learning.rate, lower = 0, upper = 1, len = 1)
      self$response = data[[target]]
      self$data = data[, !colnames(data) %in% target, drop = FALSE]
      self$optimizer = optimizer
      self$loss = loss
      self$learning.rate = learning.rate
      self$bl.factory.list = BlearnerFactoryList$new()
    },
    addLogger = function(logger, use.as.stopper = FALSE, logger.id, ...) {
      private$l.list[[logger.id]] = logger$new(use.as.stopper = use.as.stopper, ...)
    },
    getCurrentIteration = function() {
      if (!is.null(self$model) && self$model$isTrained())
        return(length(self$model$getSelectedBaselearner()))
      else
        return(0)
    },
    addBaselearner = function(features, id, bl.factory, data.source = InMemoryData, data.target = InMemoryData, ...) {
      if (self$is.initialized)
        stop("No base-learners can be added after training is started")
      for(feature in features) {
        id = paste(feature, id, sep = ".")
        private$addSingleBl(feature, id, bl.factory, data.source, data.target, ...)
      }
    },
    train = function(iteration = 100, trace = TRUE) {
      if (!self$is.initialized) {
        if (!is.null(iteration)) {
          if ("Rcpp_IterationLogger" %in% vapply(private$l.list, class, character(1))) {
            warning("Training iterations are ignored since custom iteration logger is already defined")
          } else {
            self$addLogger(IterationLogger, TRUE, logger.id = "_iterations", iter.max = iteration)
          }
        }
        private$initializeModel()
      }
      if (!self$model$isTrained())
        self$model$train(trace)
      else {
#        current.iters = self$getCurrentIteration()
        self$model$setToIteration(iteration)
      }
    }
  ),
  private = list(
    l.list = list(),
    bl.list = list(),
    initializeModel = function() {
      self$logger.list = LoggerList$new()
      for (n in names(private$l.list)) {
        self$logger.list$registerLogger(n, private$l.list[[n]])
      }
      self$model = Compboost_internal$new(self$response, self$learning.rate,
        self$stop.if.all.stoppers.fulfilled, self$bl.factory.list, self$loss, self$logger.list, self$optimizer)
      self$is.initialized = TRUE
    },
    addSingleBl = function(feature, id, bl.factory, data.source, data.target, ...) {
      data.column = self$data[[feature]]
      if (!is.numeric(data.column))
        private$addSingleCatBl(data.column, id, bl.factory, data.source, data.target, ...)
      else
        private$addSingleNumericBl(data.column, id, bl.factory, data.source, data.target, ...)
    },
    addSingleNumericBl = function(data.column, id, bl.factory, data.source, data.target, ...) {
      private$bl.list[[id]] = list()
      private$bl.list[[id]]$source = data.source$new(as.matrix(data.column), id)
      private$bl.list[[id]]$target = data.target$new()
      private$bl.list[[id]]$factory = bl.factory$new(private$bl.list[[id]]$source, private$bl.list[[id]]$target, ...)
      self$bl.factory.list$registerFactory(private$bl.list[[id]]$factory)
    },
    addSingleCatBl = function(data.column, id, bl.factory, data.source, data.target, ...) {
      private$bl.list[[id]] = list()
      lvls = unique(data.column)
      for (lvl in lvls) {
        private$addSingleNumericBl(data.column = as.matrix(as.integer(data.column == lvl)),
          id = paste(id, lvl, sep = "."), bl.factory, data.source, data.target, ...)
      }
    }
  )
)

if (FALSE) {
  load_all()
  bl = BlearnerFactoryList$new()
  source = InMemoryData$new(as.matrix(cars[["dist"]]), "dist")
  target = InMemoryData$new()
  factory = PolynomialBlearnerFactory$new(source, target, degree = 1)
  bl$registerFactory(factory)
  ll = LoggerList$new()
  log.iter = IterationLogger$new(TRUE, 100)
  ll$registerLogger("bla", log.iter)
  loss = QuadraticLoss$new()
  o = GreedyOptimizer$new()
  xx = Compboost_internal$new(cars$speed, 0.05, TRUE, bl, loss, ll, o)
  xx$train(TRUE)
  gc()
  xx$setToIteration(500)
  gc()
  xx$setToIteration(200)
  a = xx$getLoggerData()

  load_all()
  cars$dist_cat = ifelse(cars$speed > 15, "A", "B")
  cb = Compboost$new(cars, "speed", loss = QuadraticLoss$new(10))
  cb$addBaselearner("dist", "linear", PolynomialBlearnerFactory, degree = 1)
  cb$addBaselearner("dist", "quadratic", PolynomialBlearnerFactory, degree = 2)
  cb$addBaselearner("dist", "spline", PSplineBlearnerFactory, degree = 3, knots = 10, penalty = 2, differences = 2)
  cb$addBaselearner("dist_cat", "linear", PolynomialBlearnerFactory, degree = 1)
  cb$addLogger(IterationLogger, use.as.stopper = TRUE, logger.id = "bla", iter.max = 500)
  gc()
  cb$train(NULL)
  cb$train(10)
  cb$train(200)
  gc()
  cb$train(100000)
  cb$bl.factory.list
  cb$model$getLoggerData()
  head(cb$model$getParameterMatrix()[[2]])
  cb$model$getModelF
  cb$model$getParameterMatrix()[[1]]
  }
