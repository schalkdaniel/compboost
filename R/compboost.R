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
#'
#' cboost$predict(newdata = NULL)
#'
#' cboost$risk()
#'
#' cboost$selected()
#'
#' }
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
#' \strong{For cboost$predict()}:
#' \describe{
#' \item{\code{newdata}}{[\code{data.frame()}]\cr
#' 	 Data to predict on. If \code{NULL} predictions on the training data are returned.
#' }
#' }
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
#' \item{\code{target} [\code{character(1)}]}{
#' 	 Name of the Response.
#' }
#' \item{\code{id} [\code{character(1)}]}{
#' 	 Value to identify the data. By default name of \code{data}, but can be overwritten.
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
#' \item{\code{bl.factory.list} [\code{S4 FactoryList}]}{
#'   List of all registered factories represented as \code{S4 FactoryList} class.
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
#' cboost$addBaselearner("hp", "spline", PSplineBlearner, degree = 3,
#'   knots = 10, penalty = 2, differences = 2)
#' cboost$train(1000)
NULL

#'@export
Compboost = R6::R6Class("Compboost",
	public = list(
		data = NULL,
		response = NULL,
		target = NULL,
		id = NULL,
		optimizer = NULL,
		loss = NULL,
		learning.rate = NULL,
		model = NULL,
		bl.factory.list = NULL,
		stop.if.all.stoppers.fulfilled = FALSE,
		initialize = function(data, target, optimizer = GreedyOptimizer$new(), loss, learning.rate = 0.05) {
			checkmate::assertDataFrame(data, any.missing = FALSE, min.rows = 1)
			checkmate::assertCharacter(target)
			checkmate::assertNumeric(learning.rate, lower = 0, upper = 1, len = 1)

			if (! target %in% names(data)) {
				stop ("The target ", target, " is not present within the data")
			}

      # Initialize fields:
			self$target = target
			self$id = deparse(substitute(data))
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
			if (!is.null(self$model) && self$model$isTrained()) {
				return(length(self$model$getSelectedBaselearner()))
			}	else {
				return(0)
			}
		},
		addBaselearner = function(feature, id, bl.factory, data.source = InMemoryData, data.target = InMemoryData, ...) {
			if (!is.null(self$model)) {
				stop("No base-learners can be added after training is started")
			}

			# Clear base-learners which are within the bl.list but not registered:
			idx.remove = ! names(private$bl.list) %in% self$bl.factory.list$getRegisteredFactoryNames()
			if (any(idx.remove)) {
				for (i in which(idx.remove)) {
					private$bl.list[[i]] = NULL
				}
			}

			data.columns = self$data[, feature, drop = FALSE]
			id.fac = paste(paste(feature, collapse = "_"), id, sep = "_") #USE stringi

			if (ncol(data.columns) == 1 && !is.numeric(data.columns[, 1])) {
				private$addSingleCatBl(data.columns, feature, id, id.fac, bl.factory, data.source, data.target, ...)
			}	else {
				private$addSingleNumericBl(data.columns, feature, id, id.fac, bl.factory, data.source, data.target, ...)
			}
		},
		train = function(iteration = 100, trace = TRUE) {

			if (self$bl.factory.list$getNumberOfRegisteredFactories() == 0) {
				stop("Could not train without any registered base-learner.")
			}

			checkmate::assertIntegerish(iteration, lower = 1, len = 1, null.ok = TRUE)
			checkmate::assertFlag(trace)

    	# Check if it is neccessary to add a initial iteration logger. This is not the case
    	# when the user already has add one by calling `addLogger`:
			if (is.null(self$model)) {
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
				self$model$setToIteration(iteration)
			}
			return(invisible(NULL))
		},
		prepareData = function (newdata) {
				new.source.features = unique(lapply(private$bl.list, function (x) x$feature))

				new.sources = list()
				data.names = character()

				# Remove lapply due to categorical feature handling which needs to return multiple data objects
				# at once.
				for (ns in new.source.features) {

					data.columns = newdata[, ns, drop = FALSE]

					if (ncol(data.columns) == 1 && !is.numeric(data.columns[, 1])) {

						lvls = unlist(unique(data.columns))

						# Create dummy variable for each category and use that vector as data matrix. Hence,
      			# if a categorical feature has 3 groups, then these 3 groups are added as 3 different
      			# base-learners (unbiased feature selection).
						for (lvl in lvls) {
							data.names = append(data.names, paste(ns, lvl, sep = "_"))
							new.sources = c(new.sources, InMemoryData$new(as.matrix(as.integer(data.columns == lvl)), paste(ns, lvl, sep = "_")))
						}
					} else {
						data.names = append(data.names, paste(ns, collapse = "_"))
						new.sources = c(new.sources, InMemoryData$new(as.matrix(data.columns), paste(ns, collapse = "_")))
					}
				}
				names(new.sources) = data.names
				return(new.sources)
		},
		predict = function(newdata = NULL, response = FALSE) {
			checkmate::assertDataFrame(newdata, null.ok = TRUE, min.rows = 1)
			if (is.null(newdata)) {
				return(self$model$getPrediction(response))
			} else {
				return(self$model$predict(self$prepareData(newdata), response))
			}
		},
		risk = function() {
			if(!is.null(self$model)) {
				# Return the risk + intercept, hence the current iteration + 1:
				return(self$model$getRiskVector()[seq_len(self$getCurrentIteration() + 1)])
			}
			return(NULL)
		},
		selected = function() {
			if(!is.null(self$model))
				return(self$model$getSelectedBaselearner())
			return(NULL)
		},
		print = function() {
			p = glue::glue("\n
				Componentwise Gradient Boosting\n
				Trained on {self$id} with target {self$target}
				Number of base-learners: {self$bl.factory.list$getNumberOfRegisteredFactories()}
				Learning rate: {self$learning.rate}
				Iterations: {self$getCurrentIteration()}
				")

			if(!is.null(self$model))
				p = glue::glue(p, "\nOffset:{self$model$getOffset()}")

			print(p)
			print(self$loss)
		},
		coef = function () {
			if(!is.null(self$model)) {
				return(c(self$model$getEstimatedParameter(), offset = self$model$getOffset()))
			}
			return(NULL)
		},
		plot = function (blearner.type = NULL, iters = NULL, from = NULL, to = NULL, length.out = 1000) {

			if (is.null(self$model)) {
				stop("Model needs to be trained first.")
			}
			checkmate::assertIntegerish(iters, min.len = 1, any.missing = FALSE, null.ok = TRUE)
			checkmate::assertCharacter(blearner.type, len = 1, null.ok = TRUE)

			if (is.null(blearner.type)) {
				stop("Please specify a valid base-learner plus feature.")
			}
			if (! blearner.type %in% names(private$bl.list)) {
				stop("Your requested feature plus learner is not available. Check 'getFactoryNames()' for available learners.")
			}
			if (length(private$bl.list[[blearner.type]]$feature) > 1) {
				stop("Only univariate plotting is supported.")
			}
			# Check if selected base-learner includes the proposed one + check if iters is big enough:
			iter.min = which(self$selected() == blearner.type)[1]
			if (! blearner.type %in% unique(self$selected())) {
				stop("Requested base-learner plus feature was not selected.")
			} else {
				if (any(iters < iter.min)) {
					warning("Requested base-learner plus feature was first selected at iteration ", iter.min)
				}
			}
			feat.name = private$bl.list[[blearner.type]]$target$getIdentifier()

			checkmate::assertNumeric(x = self$data[[feat.name]], min.len = 2, null.ok = FALSE)
			checkmate::assertNumeric(from, lower =  min(self$data[[feat.name]]), upper = max(self$data[[feat.name]]), len = 1, null.ok = TRUE)
			checkmate::assertNumeric(to, lower =  min(self$data[[feat.name]]), upper = max(self$data[[feat.name]]), len = 1, null.ok = TRUE)

			if (is.null(from)) { 				
				from = min(self$data[[feat.name]])
			}
			if (is.null(to)) {
				to = max(self$data[[feat.name]])
			}
			if (from >= to) {
				warning("Argument from is smaller than to, hence the x interval is [to, from].")
				temp = from
				from = to
				to = temp
			}

			plot.data = as.matrix(seq(from = from, to = to, length.out = length.out))
			feat.map  = private$bl.list[[blearner.type]]$factory$transformData(plot.data)

			# Create data.frame for plotting depending if iters is specified:
			if (!is.null(iters[1])) {
				preds = lapply(iters, function (x) {
					if (x >= iter.min) {
						return(feat.map %*% self$model$getParameterAtIteration(x)[[blearner.type]])
					} else {
						return(rep(0, length.out))
					}
				})
				names(preds) = iters

				df.plot = data.frame(
					effect    = unlist(preds),
					iteration = as.factor(rep(iters, each = length.out)),
					feature   = plot.data
					)

				gg = ggplot(df.plot, aes(feature, effect, color = iteration))

			} else {
				df.plot = data.frame(
					effect  = feat.map %*% self$coef()[[blearner.type]],
					feature = plot.data
					)

				gg = ggplot(df.plot, aes(feature, effect))
			}

			gg = gg + 
			geom_line() + 
			geom_rug(data = self$data, aes_string(x = feat.name), inherit.aes = FALSE, 
				alpha = 0.8) + 
			xlab(feat.name) + 
			xlim(from, to) +
			ylab("Additive Contribution")

			return(gg)
		},
		getFactoryNames = function () {
			# return(lapply(private$bl.list, function (bl) bl[[1]]$target$getIdentifier()))
			return(names(private$bl.list))
		}
		),
private = list(
  	# Lists of single logger and base-learner factories. Neccessary to prevent the factories from the
  	# arbage collector which deallocates all the data from the heap and couses R to crash.
	l.list = list(),
	bl.list = list(),
	logger.list = list(),

	initializeModel = function() {

		private$logger.list = LoggerList$new()
		for (n in names(private$l.list)) {
			private$logger.list$registerLogger(n, private$l.list[[n]])
		}
		self$model = Compboost_internal$new(self$response, self$learning.rate,
			self$stop.if.all.stoppers.fulfilled, self$bl.factory.list, self$loss, private$logger.list, self$optimizer)
	},
	addSingleNumericBl = function(data.columns, feature, id.fac, id, bl.factory, data.source, data.target, ...) {

		private$bl.list[[id]] = list()
		private$bl.list[[id]]$source = data.source$new(as.matrix(data.columns), paste(feature, collapse = "_"))
		private$bl.list[[id]]$feature = feature
		private$bl.list[[id]]$target = data.target$new()
		private$bl.list[[id]]$factory = bl.factory$new(private$bl.list[[id]]$source, private$bl.list[[id]]$target, id.fac, ...)
		self$bl.factory.list$registerFactory(private$bl.list[[id]]$factory)
		private$bl.list[[id]]$source = NULL

	},	
	addSingleCatBl = function(data.column, feature, id.fac, id, bl.factory, data.source, data.target, ...) {

		lvls = unlist(unique(data.column))

			# Create dummy variable for each category and use that vector as data matrix. Hence,
      # if a categorical feature has 3 groups, then these 3 groups are added as 3 different
      # base-learners (unbiased feature selection).
		for (lvl in lvls) {

			list.id = paste(feature, lvl, id.fac, sep = "_")

			private$addSingleNumericBl(data.columns = as.matrix(as.integer(data.column == lvl)),
				feature = paste(feature, lvl, sep = "_"), id.fac = id.fac, 
				id = list.id, bl.factory, data.source, data.target, ...)

				# This is important because of:
				#   1. feature in addSingleNumericBl needs to be something like cat_feature_Group1 to define the
				#      data objects correctly in a unique way.
				#   2. The feature itself should not be named with the level. Instead of that we just want the
				#      feature name of the categorical variable, such as cat_feature (important for predictions).
			private$bl.list[[list.id]]$feature = feature
		}
	}
	)
)

if (FALSE) {
	load_all()
	cars$dist_cat = ifelse(cars$speed > 15, "A", "B")
	cars$foo_1 = rnorm(50)
	cb = Compboost$new(cars, "speed", loss = QuadraticLoss$new(10))
	cb
	cb$risk()
	cb$selected()
	cb$addBaselearner("dist_cat", "linear", PolynomialBlearnerFactory, degree = 1, intercept = TRUE)
	lapply(c("dist", "foo_1"), function(x) cb$addBaselearner(x, "linear", PolynomialBlearnerFactory, degree = 1))
	cb$train(5)
	cb$risk()
	cb$selected()
#	cb$addBaselearner(c("dist", "foo"), "quadratic", PolynomialBlearnerFactory, degree = 2, intercept = TRUE)
 # cb$addBaselearner("dist", "spline", PSplineBlearnerFactory, degree = 3, knots = 10, penalty = 2, differences = 2)
  #cb$addBaselearner("dist_cat", "linear", PolynomialBlearnerFactory, degree = 1)
	cb$addLogger(IterationLogger, use.as.stopper = TRUE, logger.id = "bla", iter.max = 500)
	cb$train(NULL)
	cb
	cb$predict()
	cb$predict(cars)
	cb$train(10)
	cb2$train(200)
	gc()
	cb$train(100000)
	cb$bl.factory.list
	cb$model$getLoggerData()
	head(cb$model$getParameterMatrix()[[2]])
	cb$model$getParameterMatrix()[[1]]
}
