#' Compboost API
#' 
#' \code{Compboost} wraps the \code{S4} class system exposed by \code{Rcpp} to make defining
#' objects, adding objects, the training and taking predictions, and plotting much easier.
#' As alredy mentioned, the \code{Compboost} class is just a wrapper and hence compatible
#' with the most \code{S4} classes. This together defines the compboost API.
#' 
#' @format \code{\link{R6Class}} object.
#' @name Compboost
#' @section Usage:
#' \preformatted{
#' cboost = Compboost$new(data, target, optimizer = GreedyOptimizer$new(), loss, 
#'   learning.rate = 0.05)
#' 
#' cboost$addLogger(logger, use.as.stopper = FALSE, logger.id, ...)
#' 
#' cbboost$addBaselearner(features, id, bl.factory, data.source = InMemoryData, 
#'   data.target = InMemoryData, ...)
#' 
#' cbboost$train(iteration = 100, trace = TRUE)
#' 
#' cboost$getCurrentIteration()
#' }
#' 
#' @section Arguments: 
#' \strong{For Compboost$new()}:
#' \describe{
#' \item{\code{data}}{[\code{data.frame}]\cr
#'   Data used for training.
#' }
#' \item{\code{target}}{[\code{character(1)}]\cr
#'   Character naming the target. It is necessery that target is available as column in data. 
#' }
#' \item{\code{optimizer}}{[\code{S4 Optimizer}]\cr
#'   Optimizer used for the fitting process given as initialized \code{S4 Optimizer} class. 
#'   Default is the \code{GreedyOptimizer}. 
#' }
#' \item{\code{loss}}{[\code{S4 Loss}]\cr
#'   Loss as initialized \code{S4 Loss} which is used to calculate pseudo residuals and the 
#'   empirical risk. Note that the loss needs match the data type of the target variable. 
#'   See the details for possible choices.
#' }
#' \item{\code{learning.rage}}{[\code{numeric(1)}]\cr
#'   Learning rate used to shrink estimated parameter in each iteration. The learning rate
#'   remains constant during the training and has to be between 0 and 1.
#' } 
#' }
#' 
#' \strong{For cboost$addLogger()}:
#' \describe{
#' \item{\code{logger}}{[\code{S4 Logger}]\cr
#'   Logger which are registered within a logger list. The objects must be given as uninitialized
#'   \code{S4 Logger} class. See the details for possible choices. 
#' }
#' \item{\code{use.as.stopper}}{[\code{logical(1)}]\cr
#'   Logical indicating whether the new logger should also be used as stopper. Default value is
#'   \code{FALSE}.
#' }
#' \item{\code{logger.id}}{[\code{character(1)}]\cr
#'   Id of the new logger. This is neccessary to e.g. register multiple risk logger. 
#' }
#' \item{}{\code{...}\cr
#'   Further arguments passed to the constructor of the \code{S4 Logger} class specified in 
#'   \code{logger}. For possible arguments see details or the help pages (e.g. \code{?IterationLogger}) 
#'   of the \code{S4} classes.
#' }
#' }
#' 
#' \strong{For cboost$addBaselearner()}:
#' \describe{
#' \item{\code{features}}{[\code{vector(mode = "character")}]\cr
#'   Vector of column names which are used within the specified base-learner. Each column is defined as 
#'   new base-learner by using the learner given by \code{bl.factory}.
#' }
#' \item{\code{id}}{[\code{character(1)}]\cr
#'   Id of the base-learners. This is necessry to define multiple learners with the same underlying data.
#' }
#' \item{\code{bl.factory}}{[\code{S4 Factory}]\cr
#'   Uninitialized base-learner factory represented as \code{S4 Factory} class. See the details
#'   for possible choices.
#' }
#' \item{\code{data.source}}{[\code{S4 Data}]\cr
#'   Data source object. At the moment just in memory is supported.
#' }
#' \item{\code{bl.factory}}{[\code{S4 Data}]\cr
#'   Data target object. At the moment just in memory is supported.
#' }
#' \item{}{\code{...}\cr
#'   Further arguments passed to the constructor of the \code{S4 Factory} class specified in 
#'   \code{bl.factory}. For possible arguments see the help pages (e.g. \code{?PSplineBlearnerFactory}) 
#'   of the \code{S4} classes.
#' }
#' }
#' 
#' \strong{For cboost$train()}:
#' \describe{
#' \item{\code{iteration}}{[\code{integer(1)}]\cr
#'   Set the algorithm at \code{iteration}. Note: This argument is ignored if this is the first 
#'   training and an iteration logger is already specified. For further uses the algorithm automatically
#'   continues training if \code{iteration} is set to an value larger than the already trained iterations.
#' }
#' \item{\code{trace}}{[\code{logical(1)}]\cr
#'   Logical indicating whether the trace during the fitting process should be printed or not.
#' }
#' }
#' 
#' @section Details:
#'   \strong{Loss}\cr
#'   Available choices for the loss are:
#' 	 \itemize{
#'   \item 
#'     \code{QuadraticLoss} (Regression)
#' 
#'   \item 
#'     \code{AbsoluteLoss} (Regression)
#' 
#'   \item 
#'     \code{BinomialLoss} (Binary Classification)
#' 
#'   \item 
#'     \code{CustomLoss} (Custom)
#' 
#'   \item 
#'     \code{CustomCppLoss} (Custom) 
#'   }
#'   (For each loss also take a look at the help pages (e.g. \code{?BinomialLoss}) and the
#'   \code{C++} documentation for details about the underlying formulas)
#' 	 
#'   \strong{Logger}\cr
#'   Available choices for the logger are:
#'   \itemize{
#'   \item 
#'     \code{IterationLogger}: Log current iteration. Additional arguments:
#'     \describe{
#'       \item{\code{max_iterations} [\code{integer(1)}]}{
#'         Maximal number of iterations.
#'       }
#'     }
#' 
#'   \item 
#'     \code{TimeLogger}: Log already ellapsed time. Additional arguments:
#'     \describe{
#'       \item{\code{max_time} [\code{integer(1)}]}{
#'         Maximal time for the computation.
#'       }
#'       \item{\code{time_unit} [\code{character(1)}]}{
#'         Character to specify the time unit. Possible choices are \code{minutes}, \code{seconds}, or \code{microseconds}.
#'       }
#'     }
#' 
#'   \item 
#'     \code{InbagRiskLogger}:
#'     \describe{
#'       \item{\code{used_loss} [\code{S4 Loss}]}{
#'         Loss as initialized \code{S4 Loss} which is used to calculate the empirical risk. See the 
#'         details for possible choices.
#'       }
#'       \item{\code{eps_for_break} [\code{numeric(1)}]}{
#'         This argument is used if the logger is also used as stopper. If the relative improvement 
#'         of the logged inbag risk falls above this boundary the stopper breaks the algorithm.
#'       }
#'     }
#' 
#'   \item 
#'     \code{OobRiskLogger}:
#'     \describe{
#'       \item{\code{used_loss} [\code{S4 Loss}]}{
#'         Loss as initialized \code{S4 Loss} which is used to calculate the empirical risk. See the 
#'         details for possible choices.
#'       }
#'       \item{\code{eps_for_break} [\code{numeric(1)}]}{
#'         This argument is used if the logger is also used as stopper. If the relative improvement 
#'         of the logged inbag risk falls above this boundary the stopper breaks the algorithm.
#'       }
#'       \item{\code{oob_data} [\code{list}]}{
#'         A list which contains data source objects which corresponds to the source data of each registered factory. 
#'         The source data objects should contain the out of bag data. This data is then used to calculate the 
#'         new predictions in each iteration.
#'       }
#'       \item{\code{oob_response} [\code{vector}]}{
#'         Vector which contains the response for the out of bag data given within \code{oob_data}.
#'       }
#'     }
#'   }
#'   \strong{Note}: 
#'   \itemize{
#'   \item 
#'     Even if you do not use the logger as stopper you have to define the arguments such as \code{max_time}. 
#'   
#'   \item 
#'     We are aware of that the style guide here is not consistent with the \code{R6} arguments. Nevertheless, using
#'     \code{_} as word seperator is due to the used arguments within \code{C++}.
#'   }
#' 
#' @section Fields:
#' \describe{
#' \item{\code{data} [\code{data.frame}]}{
#'   Data used for training the algorithm.
#' }
#' \item{\code{response} [\code{vector}]}{
#'   Response given as vector.
#' }
#' \item{\code{optimizer} [\code{S4 Optimizer}]}{
#'   Optimizer used within the fitting process.
#' }
#' \item{\code{loss} [\code{S4 Loss}]}{
#'   Loss used to calculate pseudo residuals and empirical risk.
#' }
#' \item{\code{learning.rate} [\code{numeric(1)}]}{
#'   Learning rate used to shrink the estimated parameter in each iteration.
#' }
#' \item{\code{model} [\code{S4 Compboost_internal}]}{
#'   Internal \code{S4 Compboost_internal} class on which the main operations are called.
#' }
#' \item{\code{is.initialized} [\code{logical(1)}]}{
#'   Logical indicating if the algorithm is already initialized (Be aware that initialized does
#'   not mean trained).
#' }
#' \item{\code{bl.factory.list} [\code{S4 FactoryList}]}{
#'   List of all registered factories represented as \code{S4 FactoryList} class.
#' }
#' \item{\code{logger.list} [\code{S4 LoggerList}]}{
#'   List of all registered logger represented as \code{S4 LoggerList} class.
#' }
#' \item{\code{stop.if.all.stoppers.fulfilled} [\code{logical(1)}]}{
#'   Logical indicating whether all stopper should be used symultaniously or if it is sufficient
#'   that the first stopper which is fulfilled breaks the algorithm.
#' }
#' }
#' 
#' @section Methods:
#' \describe{
#' \item{\code{addLogger}}{method to add a logger to the algorithm (Note: This is just possible before the training).}
#' \item{\code{getCurrentIteration}}{method to get the current iteration on which the algorithm is set.}
#' \item{\code{addBaselearner}}{method to add a new base-learner factories to the algorithm (Note: This is just possible before the training).}
#' \item{\code{train}}{method train the algorithm.}
#' \item{\code{initializeModel}}{[internal] method to initialize the \code{Compboost} object}
#' \item{\code{addSingleBl}}{[internal] method to add a single factory to the factory list.}
#' \item{\code{AddSingleNumericBl}}{[internal] method to add a single factory list for a numerical feature.}
#' \item{\code{AddSingleCatBl}}{[internal] method to add a single factory list for a categorical feature.}
#' }
#' 
#' @examples 
#' cboost = Compboost$new(mtcars, "mpg", loss = QuadraticLoss$new())
#' cboost$addBaselearner(c("hp", "wt"), "spline", PSplineBlearnerFactory, degree = 3, 
#'   knots = 10, penalty = 2, differences = 2)
#' cboost$train(1000)
NULL

#'@export
Compboost = R6::R6Class("Compboost",
	public = list(

  	# Public fields:
		data = NULL,
		response = NULL, # Should be private?
		optimizer = NULL, # Should be private?
		loss = NULL, # Should be private?
		learning.rate = NULL,
		model = NULL,
		is.initialized = FALSE, # Should be private?
		bl.factory.list = NULL,
		logger.list = list(),
		stop.if.all.stoppers.fulfilled = FALSE,
		initialize = function(data, target, optimizer = GreedyOptimizer$new(), loss, learning.rate = 0.05) {

			checkmate::assertDataFrame(data, any.missing = FALSE, min.rows = 1)
			checkmate::assertCharacter(target)
			checkmate::assertNumeric(learning.rate, lower = 0, upper = 1, len = 1)

			if (! target %in% names(data)) {
				stop ("The target ", target, " is not present within the data")
			}

      # Initialize fields:
			self$response = data[[target]]
			self$data = data[, !colnames(data) %in% target, drop = FALSE]
			self$optimizer = optimizer
			self$loss = loss
			self$learning.rate = learning.rate

      # Initialize new base-learner factory list. All factories which are defined in 
      # `addBaselearners` are registered here:
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
      # if (!is.list(features))
      #   features = list(features)

      # Define and add for each feature in `features` a new base-learner:
			for(feature in features) {
				id.feat = paste(feature, id, sep = ".")
				private$addSingleBl(feature, id.feat, bl.factory, data.source, data.target, ...)
			}
		},
		train = function(iteration = 100, trace = TRUE) {

			checkmate::checkInteger(iteration, lower = 0, len = 1, null.ok = TRUE)

    	# Check if it is neccessary to add a initial iteration logger. This is not the case
    	# when the user already has add one by calling `addLogger`:
			if (!self$is.initialized) {
      	# If iteration is NULL, then there is no new iteration logger defined. This could be
      	# used, for example, to train the algorithm an break it after a defined number of 
      	# hours or minutes.
				if (!is.null(iteration)) {
        	# Add new logger in the case that there isn't already a custom defined one:
					if ("Rcpp_IterationLogger" %in% vapply(private$l.list, class, character(1))) {
						warning("Training iterations are ignored since custom iteration logger is already defined")
					} else {
						self$addLogger(IterationLogger, TRUE, logger.id = "_iterations", iter.max = iteration)
					}
				}
        # After calling `initializeModel` it isn't possible to add base-learner or logger.
				private$initializeModel()
			}
      # Just call train for the initial fitting process. If the model is alredy initially trained,
      # then we use `setToIteration` to set to a lower iteration or retrain the model.
			if (!self$model$isTrained())
				self$model$train(trace)
			else {
#        current.iters = self$getCurrentIteration()
				self$model$setToIteration(iteration)
			}
		}
		),
	private = list(
  	# Lists of single logger and base-learner factories. Neccessary to prevent the factories from the 
  	# arbage collector which deallocates all the data from the heap and couses R to crash.
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
			private$bl.list[[id]]$source = NULL

		},
		addSingleCatBl = function(data.column, id, bl.factory, data.source, data.target, ...) {
			private$bl.list[[id]] = list()
			lvls = unique(data.column)
      # Create dummy variable for each category and use that vector as data matrix. Hence, 
      # if a categorical feature has 3 groups, then these 3 groups are added as 3 different
      # base-learners (unbiased feature selection).
			for (lvl in lvls) {
				private$addSingleNumericBl(data.column = as.matrix(as.integer(data.column == lvl)),
					id = paste(id, lvl, sep = "."), bl.factory, data.source, data.target, ...)
			}
		}
		)
	)

# if (FALSE) {
#   load_all()
#   bl = BlearnerFactoryList$new()
#   source = InMemoryData$new(as.matrix(cars[["dist"]]), "dist")
#   target = InMemoryData$new()
#   factory = PolynomialBlearnerFactory$new(source, target, degree = 1)
#   bl$registerFactory(factory)
#   ll = LoggerList$new()
#   log.iter = IterationLogger$new(TRUE, 100)
#   ll$registerLogger("bla", log.iter)
#   loss = QuadraticLoss$new()
#   o = GreedyOptimizer$new()
#   xx = Compboost_internal$new(cars$speed, 0.05, TRUE, bl, loss, ll, o)
#   xx$train(TRUE)
#   gc()
#   xx$setToIteration(500)
#   gc()
#   xx$setToIteration(200)
#   a = xx$getLoggerData()

#   load_all()
#   cars$dist_cat = ifelse(cars$speed > 15, "A", "B")
#   cars$foo = rnorm(50)
#   cb = Compboost$new(cars, "speed", loss = QuadraticLoss$new(10))
#   cb$addBaselearner(c("dist", "foo"), "linear", PolynomialBlearnerFactory, degree = 1)
#   cb$addBaselearner("dist", "quadratic", PolynomialBlearnerFactory, degree = 2)
#   cb$addBaselearner("dist", "spline", PSplineBlearnerFactory, degree = 3, knots = 10, penalty = 2, differences = 2)
#   cb$addBaselearner("dist_cat", "linear", PolynomialBlearnerFactory, degree = 1)
#   cb$addLogger(IterationLogger, use.as.stopper = TRUE, logger.id = "bla", iter.max = 500)
#   gc()
#   cb$train(NULL)
#   cb$train(10)
#   cb$train(200)
#   gc()
#   cb$train(100000)
#   cb$bl.factory.list
#   cb$model$getLoggerData()
#   head(cb$model$getParameterMatrix()[[2]])
#   cb$model$getModelF
#   cb$model$getParameterMatrix()[[1]]
# }
