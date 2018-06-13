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
    addBaseLearner = function(feature, bl.factory, data.source = InMemoryData, data.target = InMemoryData, ...) {
      if (self$is.initialized)
        stop("No base-learners can be added after training is started")
      source = data.source$new(as.matrix(self$data[[feature]]), feature)
      target = data.target$new()
      factory = bl.factory$new(source, target, ...)
      self$bl.factory.list$registerFactory(factory)
    },
    addLogger = function(logger, use.as.stopper = FALSE, logger.id, ...) {
      self$logger.list[[logger.id]] = logger$new(use.as.stopper = use.as.stopper, ...)
    },
    train = function(iteration = 100, trace = TRUE) {
      if (!self$is.initialized) {
        if (!is.null(iteration)) {
          if ("Rcpp_IterationLogger" %in% vapply(self$logger.list, class, character(1))) {
            warning("Training iterations are ignored since custom iteration logger is already defined")
          } else {
            self$addLogger(IterationLogger, TRUE, logger.id = "_iterations", iter.max = iteration)
          }
        }
        private$initializeModel()
      }
      if (!self$model$isTrained())
        self$model$train(trace)
      else
        self$model$setToIteration(iteration)
    }
  ),
  private = list(
    initializeModel = function() {
      ll = LoggerList$new()
      mapply(function(n, l) ll$registerLogger(n, l), n = names(self$logger.list), l = self$logger.list)
      self$model = Compboost_internal$new(self$response, self$learning.rate,
        self$stop.if.all.stoppers.fulfilled, self$bl.factory.list, self$loss, ll, self$optimizer)
      self$is.initialized = TRUE
    }
  )
)

if (FALSE) {
  cb = Compboost$new(cars, "speed", loss = QuadraticLoss$new())
  cb$addBaseLearner("dist", PolynomialBlearnerFactory, degree = 1)
  cb$addBaseLearner("dist", PolynomialBlearnerFactory, degree = 2)
  cb$addBaseLearner("dist", PSplineBlearnerFactory, degree = 3, knots = 10, penalty = 2, differences = 2)
  cb$addLogger(IterationLogger, use.as.stopper = TRUE, logger.id = "bla", iter.max = 100)
  cb$train(100)
}
