#' Compboost API
#'
#' \code{Compboost} wraps the \code{S4} class system exposed by \code{Rcpp} to make defining
#' objects, adding objects, the training, calculating predictions, and plotting much easier.
#' As already mentioned, the \code{Compboost R6} class is just a wrapper and compatible
#' with the most \code{S4} classes.
#'
#' @format \code{\link{R6Class}} object.
#' @name Compboost
#' @section Usage:
#' \preformatted{
#' cboost = Compboost$new(data, target, optimizer = OptimizerCoordinateDescent$new(), loss,
#'   learning_rate = 0.05, oob_fraction)
#'
#' cboost$addLogger(logger, use_as_stopper = FALSE, logger_id, ...)
#'
#' cbboost$addBaselearner(features, id, bl_factory, data_source = InMemoryData,
#'   data_target = InMemoryData, ...)
#'
#' cbboost$train(iteration = 100, trace = -1)
#'
#' cboost$getCurrentIteration()
#'
#' cboost$predict(newdata = NULL)
#'
#' cboost$getInbagRisk()
#'
#' cboost$getSelectedBaselearner()
#'
#' cboost$getEstimatedCoef()
#'
#' cboost$plot(blearner_name = NULL, iters = NULL, from = NULL, to = NULL, length_out = 1000)
#'
#' cboost$getBaselearnerNames()
#'
#' cboost$prepareData(newdata)
#'
#' cboost$getLoggerData()
#'
#' cboost$calculateFeatureImportance(num_feats = NULL)
#'
#' cboost$plotFeatureImportance(num_feats = NULL)
#'
#' cboost$plotInbagVsOobRisk()
#'
#' }
#' @section Arguments:
#' \strong{For Compboost$new()}:
#' \describe{
#'   \item{\code{data}}{[\code{data.frame}]\cr
#'     A data frame containing the data.
#'   }
#'   \item{\code{target}}{[\code{character(1)}]\cr
#'     Character value containing the target variable. Note that the loss must match the
#'     data type of the target.
#'   }
#'   \item{\code{optimizer}}{[\code{S4 Optimizer}]\cr
#'     An initialized \code{S4 Optimizer} object exposed by Rcpp (e.g. \code{OptimizerCoordinateDescent$new()})
#'     to select features at each iteration.
#'   }
#'   \item{\code{loss}}{[\code{S4 Loss}]\cr
#'     Initialized \code{S4 Loss} object exposed by Rcpp that is used to calculate the risk and pseudo
#'     residuals (e.g. \code{LossQuadratic$new()}).
#'   }
#'   \item{\code{learning.rage}}{[\code{numeric(1)}]\cr
#'     Learning rate to shrink the parameter in each step.
#'   }
#'   \item{\code{oob_fraction}}{[\code{numeric(1)}]\cr
#'     Fraction of how much data are used to track the out of bag risk.
#'   }
#' }
#'
#' \strong{For cboost$addLogger()}:
#' \describe{
#'   \item{\code{logger}}{[\code{S4 Logger}]\cr
#'     Uninitialized \code{S4 Logger} class object that is registered in the model.
#'     See the details for possible choices.
#'   }
#'   \item{\code{use_as_stopper}}{[\code{logical(1)}]\cr
#'     Logical value indicating whether the new logger should also be used as stopper
#'     (early stopping). Default value is \code{FALSE}.
#'   }
#'   \item{\code{logger_id}}{[\code{character(1)}]\cr
#'     Id of the new logger. This is necessary to, for example, register multiple risk logger.
#'   }
#'   \item{}{\code{...}\cr
#'     Further arguments passed to the constructor of the \code{S4 Logger} class specified in
#'     \code{logger}. For possible arguments see details or the help pages (e.g. \code{?LoggerIteration}).
#'   }
#' }
#'
#' \strong{For cboost$addBaselearner()}:
#' \describe{
#' \item{\code{features}}{[\code{character()}]\cr
#'   Vector of column names which are used as input data matrix for a single base-learner. Note that not
#'   every base-learner supports the use of multiple features (e.g. the spline base-learner does not).
#' }
#' \item{\code{id}}{[\code{character(1)}]\cr
#'   Id of the base-learners. This is necessary since it is possible to define multiple learners with the same underlying data.
#' }
#' \item{\code{bl_factory}}{[\code{S4 Factory}]\cr
#'   Uninitialized base-learner factory given as \code{S4 Factory} class. See the details
#'   for possible choices.
#' }
#' \item{\code{data_source}}{[\code{S4 Data}]\cr
#'   Data source object. At the moment just in memory is supported.
#' }
#' \item{\code{data_target}}{[\code{S4 Data}]\cr
#'   Data target object. At the moment just in memory is supported.
#' }
#' \item{}{\code{...}\cr
#'   Further arguments passed to the constructor of the \code{S4 Factory} class specified in
#'   \code{bl_factory}. For possible arguments see the help pages (e.g. \code{?BaselearnerPSplineFactory})
#'   of the \code{S4} classes.
#' }
#' }
#'
#' \strong{For cboost$train()}:
#' \describe{
#'   \item{\code{iteration}}{[\code{integer(1)}]\cr
#'     Number of iterations that are trained. If the model is already trained the model is set to the given number
#'     by goint back through the already trained base-learners or training new ones. Note: This function defines an
#'     iteration logger with the id \code{_iterations} which is then used as stopper.
#'   }
#'   \item{\code{trace}}{[\code{integer(1)}]\cr
#'     Integer indicating how often a trace should be printed. Specifying \code{trace = 10}, then every
#'     10th iteration is printed. If no trace should be printed set \code{trace = 0}. Default is
#'     -1 which means that in total 40 iterations are printed.
#'   }
#' }
#'
#' \strong{For cboost$predict()}:
#' \describe{
#' \item{\code{newdata}}{[\code{data.frame()}]\cr
#' 	 Data to predict on. If newdata equals \code{NULL} predictions on the training data are returned.
#' }
#' }
#' \strong{For cboost$plot()}:
#' \describe{
#' \item{\code{blearner_name}}{[\code{character(1)}]\cr
#' 	 Character name of the base-learner to plot the additional contribution to the response.
#' }
#' \item{\code{iters}}{[\code{integer()}]\cr
#' 	 Integer vector containing the iterations the user wants to illustrate.
#' }
#' \item{\code{from}}{[\code{numeric(1)}]\cr
#' 	 Lower bound for plotting (should be smaller than \code{to}).
#' }
#' \item{\code{to}}{[\code{numeric(1)}]\cr
#' 	 Upper bound for plotting (should be greater than \code{from}).
#' }
#' \item{\code{length_out}}{[\code{integer(1)}]\cr
#' 	 Number of equidistant points between \code{from} and \code{to} used for plotting.
#' }
#' }
#' @section Details:
#'   \strong{Loss}\cr
#'   Available choices for the loss are:
#' 	 \itemize{
#'     \item
#'       \code{LossQuadratic} (Regression)
#'
#'     \item
#'       \code{LossAbsolute} (Regression)
#'
#'     \item
#'       \code{LossBinomial} (Binary Classification)
#'
#'     \item
#'       \code{LossCustom} (Custom)
#'
#      \item
#        \code{LossCustomCpp} (Custom)
#'   }
#'   (For each loss take also a look at the help pages (e.g. \code{?LossBinomial}) and the
#'   \code{C++} documentation for details)
#'
#'   \strong{Logger}\cr
#'   Available choices for the logger are:
#'   \itemize{
#'     \item
#'       \code{LoggerIteration}: Logs the current iteration. Additional arguments:
#'       \describe{
#'         \item{\code{max_iterations} [\code{integer(1)}]}{
#'           Maximal number of iterations.
#'         }
#'       }
#'
#'     \item
#'       \code{LoggerTime}: Logs the elapsed time. Additional arguments:
#'       \describe{
#'         \item{\code{max_time} [\code{integer(1)}]}{
#'           Maximal time for the computation.
#'         }
#'         \item{\code{time_unit} [\code{character(1)}]}{
#'           Character to specify the time unit. Possible choices are \code{minutes}, \code{seconds}, or \code{microseconds}.
#'         }
#'       }
#'
#'     \item
#'       \code{LoggerInbagRisk}:
#'       \describe{
#'         \item{\code{used_loss} [\code{S4 Loss}]}{
#'           Loss as initialized \code{S4 Loss} which is used to calculate the empirical risk. See the
#'           details for possible choices.
#'         }
#'         \item{\code{eps_for_break} [\code{numeric(1)}]}{
#'           This argument is used if the logger is also used as stopper. If the relative improvement
#'           of the logged inbag risk falls below this boundary, then the stopper breaks the algorithm.
#'         }
#'       }
#'
#'     \item
#'       \code{LoggerOobRisk}:
#'       \describe{
#'         \item{\code{used_loss} [\code{S4 Loss}]}{
#'           Loss as initialized \code{S4 Loss} which is used to calculate the empirical risk. See the
#'           details for possible choices.
#'         }
#'         \item{\code{eps_for_break} [\code{numeric(1)}]}{
#'           This argument is used if the logger is also used as stopper. If the relative improvement
#'           of the logged inbag risk falls above this boundary the stopper breaks the algorithm.
#'         }
#'         \item{\code{oob_data} [\code{list}]}{
#'           A list which contains data source objects which corresponds to the source data of each registered factory.
#'           The source data objects should contain the out of bag data. This data is then used to calculate the
#'           new predictions in each iteration.
#'         }
#'         \item{\code{oob_response} [\code{vector}]}{
#'           Vector which contains the response for the out of bag data given within \code{oob_data}.
#'         }
#'       }
#'     }
#'
#'   \strong{Note}:
#'   \itemize{
#'     \item
#'       Even if you do not use the logger as stopper you have to define the arguments such as \code{max_time}.
#'
#'     \item
#'       We are aware of that the style guide here is not consistent with the \code{R6} arguments. Nevertheless, using
#'       \code{_} as word separator is due to the used argument names within \code{C++}.
#'   }
#'
#' @section Fields:
#' \describe{
#'   \item{\code{data} [\code{data.frame}]}{
#'     Data used for training the algorithm.
#'   }
#'   \item{\code{data_oob} [\code{data.frame}]}{
#'     Data used for out of bag tracking.
#'   }
#'   \item{\code{oob_fraction} [\code{numeric(1)}]}{
#'     Fraction of how much data are used to track the out of bag risk.
#'   }
#'   \item{\code{response} [\code{vector}]}{
#'     Response vector.
#'   }
#'   \item{\code{target} [\code{character(1)}]}{
#'   	 Name of the target variable
#'   }
#'   \item{\code{id} [\code{character(1)}]}{
#'   	 Name of the given dataset.
#'   }
#'   \item{\code{optimizer} [\code{S4 Optimizer}]}{
#'     Optimizer used within the fitting process.
#'   }
#'   \item{\code{loss} [\code{S4 Loss}]}{
#'     Loss used to calculate pseudo residuals and empirical risk.
#'   }
#'   \item{\code{learning_rate} [\code{numeric(1)}]}{
#'     Learning rate used to shrink the estimated parameter in each iteration.
#'   }
#'   \item{\code{model} [\code{S4 Compboost_internal}]}{
#'     \code{S4 Compboost_internal} class object from that the main operations are called.
#'   }
#'   \item{\code{bl_factory_list} [\code{S4 FactoryList}]}{
#'     List of all registered factories represented as \code{S4 FactoryList} class.
#'   }
#'   \item{\code{positive_category} [\code{character(1)}]}{
#'     Character containing the name of the positive class in the case of (binary) classification.
#'   }
#'   \item{\code{stop_if_all_stoppers_fulfilled} [\code{logical(1)}]}{
#'     Logical indicating whether all stopper should be used simultaneously or if it is sufficient
#'     to just use the first stopper to stop the algorithm.
#'   }
#' }
#'
#' @section Methods:
#' \describe{
#'   \item{\code{addLogger}}{method to add a logger to the algorithm (Note: This is just possible before the training).}
#'   \item{\code{addBaselearner}}{method to add a new base-learner to the algorithm (Note: This is just possible before the training).}
#'   \item{\code{getCurrentIteration}}{method to get the current iteration on which the algorithm is set.}
#'   \item{\code{train}}{method to train the algorithm.}
#'   \item{\code{predict}}{method to predict on a trained object.}
#'   \item{\code{getSelectedBaselearner}}{method to get a character vector of selected base-learner.}
#'   \item{\code{getEstimatedCoef}}{method to get a list of estimated coefficient of each selected base-learner.}
#'   \item{\code{plot}}{method to plot individual feature effects.}
#'   \item{\code{getBaselearnerNames}}{method to get the names of the registered factories.}
#'   \item{\code{prepareData}}{method to prepare data to track the out of bag risk of an arbitrary loss/performance function.}
#'   \item{\code{getLoggerData}}{method to the the logged data from all registered logger.}
#'   \item{\code{calculateFeatureImportance}}{method to calculate feature importance.}
#'   \item{\code{plotFeatureImportance}}{method to plot the feature importance calculated by \code{calulateFeatureImportance}.}
#'   \item{\code{plotInbagVsOobRisk}}{method to plot the inbag vs the out of bag behavior. This is just applicable if a logger with name \code{oob_logger} was registered. This is automatically done if the \code{oob_fraction} is set.}
#' }
#'
#' @examples
#' cboost = Compboost$new(mtcars, "mpg", loss = LossQuadratic$new(), oob_fraction = 0.3)
#' cboost$addBaselearner("hp", "spline", BaselearnerPSpline, degree = 3,
#'   n.knots = 10, penalty = 2, differences = 2)
#' cboost$train(1000)
#'
#' table(cboost$getSelectedBaselearner())
#' cboost$plot("hp_spline")
#' cboost$plotInbagVsOobRisk()
NULL

#' @export
Compboost = R6::R6Class("Compboost",
  public = list(
    data = NULL,
    data_oob = NULL,
    oob_fraction = NULL,
    response = NULL,
    response_oob = NULL,
    target = NULL,
    id = NULL,
    optimizer = NULL,
    loss = NULL,
    learning_rate = NULL,
    model = NULL,
    bl_factory_list = NULL,
    positive_category = NULL,
    stop_if_all_stoppers_fulfilled = FALSE,
    initialize = function(data, target, optimizer = OptimizerCoordinateDescent$new(), loss, learning_rate = 0.05, oob_fraction = NULL) {
      checkmate::assertDataFrame(data, any.missing = FALSE, min.rows = 1)
      checkmate::assertNumeric(learning_rate, lower = 0, upper = 1, any.missing = FALSE, len = 1)
      checkmate::assertNumeric(oob_fraction, lower = 0, upper = 1, any.missing = FALSE, len = 1, null.ok = TRUE)

      if (! isRcppClass(target, "Response")) {
        if (! target %in% names(data)) {
          stop ("The target ", target, " is not present within the data")
        } 
      }
      if (inherits(loss, "C++Class")) {
        stop ("Loss should be an initialized loss object by calling the constructor: ", deparse(substitute(loss)), "$new()")
      }

      self$id = deparse(substitute(data))
      data = droplevels(as.data.frame(data))
      
      if (is.character(target)) {
        checkmate::assertCharacter(target)
        if (! target %in% names(data))
          stop ("The target ", target, " is not present within the data")
        
        # With .vectorToRespone we are very restricted to the task types. We can just guess for regression or classification. For every
        # other task one should use the Response interface!
        self$response = vectorToResponse(data[[target]], target)
      } else {
        assertRcppClass(target, "Response")
        
        if(class(target)[1] == "Rcpp_ResponseFDA"){
          data = data[rep(seq_len(nrow(data)), each = length(target$getGrid())),]
        }
        if (nrow(target$getResponse()) != nrow(data))
          stop("Response must have same number of observations as the given dataset")
        self$response = target
      }
      
      
      if (! is.null(oob_fraction)) {
        private$oob_idx = sample(x = seq_len(nrow(data)), size = floor(oob_fraction * nrow(data)), replace = FALSE)
      }
      private$train_idx = setdiff(seq_len(nrow(data)), private$oob_idx)
      

      self$oob_fraction = oob_fraction
      self$target = self$response$getTargetName()
      self$data = data[private$train_idx, !colnames(data) %in% self$target, drop = FALSE]
      self$optimizer = optimizer
      self$loss = loss
      self$learning_rate = learning_rate
      if (! is.null(self$oob_fraction)) {
        self$data_oob = data[private$oob_idx, !colnames(data) %in% target, drop = FALSE]
        self$response_oob = vectorToResponse(data[private$oob_idx, self$target], "oob_response")
        self$response$filter(private$train_idx)
      }

      # Initialize new base-learner factory list. All factories which are defined in
      # `addBaselearners` are registered here:
      self$bl_factory_list = BlearnerFactoryList$new()

    },
    addLogger = function(logger, use_as_stopper = FALSE, logger_id, ...) {
      private$l_list[[logger_id]] = logger$new(logger_id, use_as_stopper = use_as_stopper, ...)
    },
    getCurrentIteration = function() {
      if (!is.null(self$model) && self$model$isTrained()) {
        return(length(self$model$getSelectedBaselearner()))
      }	else {
        return(0)
      }
    },
    addBaselearner = function(feature, id, bl_factory, data_source = InMemoryData, data_target = InMemoryData, ...) {
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
      
      # Check if the response functional data
      if(class(self$response) != "Rcpp_ResponseFDA"){
        data_columns = self$data[, feature, drop = FALSE]
        id_fac = paste(paste(feature, collapse = "_"), id, sep = "_") #USE stringi
  
        if (ncol(data_columns) == 1 && !is.numeric(data_columns[, 1])) {
          private$addSingleCatBl(data_columns, feature, id, id_fac, bl_factory, data_source, data_target, ...)
        }	else {
          private$addSingleNumericBl(data_columns, feature, id, id_fac, bl_factory, data_source, data_target, ...)
        }
      } else {
        
      }
    },
    CombineBaselearners = function(bl1, bl2, bl_factory = BaselearnerPolynomial, keep = FALSE){
      if(! bl1 %in% easy_boost$getBaselearnerNames()){
        stop("Bl1 is not a registered Baselearner.")
      }
      if(! bl2 %in% easy_boost$getBaselearnerNames()){
        stop("Bl2 is not a registered Baselearner.")
      }
      if(!is.logical(keep)){
        stop("keep must be logical.")
      }
      if(class(private$bl_list[[bl1]]$factory) != class(private$bl_list[[bl2]]$factory)){
        stop("Baselearners must be of the same kind to be combined.")
      }
      
      # Get Design Matrices from both learners
      bl1_design_mat = private$bl_list[[bl1]]$factory$getData()
      bl2_design_mat = private$bl_list[[bl2]]$factory$getData()
  
      bln = paste0(bl1,"_%_",bl2)
      bln_design_mat = as.matrix(rowwiseTensor(bl1_design_mat, bl2_design_mat))
      
      bln_source = InMemoryData$new(as.matrix(bln_design_mat), bln)
      bln_target = InMemoryData$new()
      
      private$bl_list[[bln]] = list()
      private$bl_list[[bln]]$source = bln_source
      private$bl_list[[bln]]$feature = bln
      private$bl_list[[bln]]$target = bln_target
      private$bl_list[[bln]]$factory = BaselearnerPolynomial$new(private$bl_list[[bln]]$source, 
                                                                 private$bl_list[[bln]]$target, 
                                                                 "", 
                                                                 list(degree = 1, intercept = FALSE))

      self$bl_factory_list$registerFactory(private$bl_list[[bln]]$factory)
      private$bl_list[[bln]]$source = NULL
      
      
      if(!keep){
        private$bl_list[[bl1]] = NULL
        private$bl_list[[bl2]] = NULL
      }
        
    },CenterBaselearner = function(bl_target, bl_center){
      if(! bl_target %in% easy_boost$getBaselearnerNames()){
        stop("bl_target is not a registered Baselearner.")
      }
      if(! bl_center %in% easy_boost$getBaselearnerNames()){
        stop("bl_center is not a registered Baselearner.")
      }
      if(class(private$bl_list[[bl_target]]$factory) != class(private$bl_list[[bl_center]]$factory)){
        stop("Baselearners must be of the same kind to be combined.")
      }
      
      # Get Design Matrices from both learners
      bl_target_design_mat = private$bl_list[[bl_target]]$factory$getData()
      bl_center_design_mat = private$bl_list[[bl_center]]$factory$getData()
      
      blc = paste0(bl_target,"_|_",bl_center)
      
      # calculate QR-decomposition
      qr_x1x2 = qr(t(bl_target_design_mat)%*%bl_center_design_mat)
      # get rank, which can be used to 
      # determine the zero part of R
      rank_x1x2 <- qr_x1x2$rank
      # get Q
      Q <- qr.Q(qr_x1x2, complete = TRUE)
      # then Z is given by:
      Z <- Q[, (rank_x1x2 + 1):ncol(Q)]
      # and the we have    
      blc_design_mat <- bl_target_design_mat %*% Z
      
      
      blc_source = InMemoryData$new(as.matrix(blc_design_mat), blc)
      blc_target = InMemoryData$new()
      
      private$bl_list[[bl_target]]$source = blc_source
      private$bl_list[[bl_target]]$feature = blc
      private$bl_list[[bl_target]]$target = blc_target
      
      private$bl_list[[blc]] = private$bl_list[[bl_target]]
      private$bl_list[[bl_target]] = NULL

    },
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
        if (! is.null(self$oob_fraction)) private$addOobLogger()
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
    prepareData = function (newdata) {
      new_source_features = unique(lapply(private$bl_list, function (x) x$feature))

      new_sources = list()
      data_names = character()

      # Remove lapply due to categorical feature handling which needs to return multiple data objects
      # at once.
      for (ns in new_source_features) {

        data_columns = newdata[, ns, drop = FALSE]

        if (ncol(data_columns) == 1 && !is.numeric(data_columns[, 1])) {

          lvls = unlist(unique(data_columns))

          # Create dummy variable for each category and use that vector as data matrix. Hence,
          # if a categorical feature has 3 groups, then these 3 groups are added as 3 different
          # base-learners (unbiased feature selection).
          for (lvl in lvls) {
            data_names = append(data_names, paste(ns, lvl, sep = "_"))
            new_sources = c(new_sources, InMemoryData$new(as.matrix(as.integer(data_columns == lvl)), paste(ns, lvl, sep = "_")))
          }
        } else {
          data_names = append(data_names, paste(ns, collapse = "_"))
          new_sources = c(new_sources, InMemoryData$new(as.matrix(data_columns), paste(ns, collapse = "_")))
        }
      }
      names(new_sources) = data_names
      return(new_sources)
    },
    predict = function(newdata = NULL, response = FALSE) {
      checkmate::assertDataFrame(newdata, null.ok = TRUE, min.rows = 1)
      if (is.null(newdata)) {
        return(self$model$getPrediction(response))
      } else {
        return(self$model$predict(self$prepareData(newdata), response))
      }
    },
    getInbagRisk = function() {
      if(!is.null(self$model)) {
        # Return the risk + intercept, hence the current iteration + 1:
        return(self$model$getRiskVector()[seq_len(self$getCurrentIteration() + 1)])
      }
      return(NULL)
    },
    getSelectedBaselearner = function() {
      if(!is.null(self$model))
        return(self$model$getSelectedBaselearner())
      return(NULL)
    },
    print = function() {
      p = glue::glue("\n
				Component-Wise Gradient Boosting\n
				Trained on {self$id} with target {self$target}
				Number of base-learners: {self$bl_factory_list$getNumberOfRegisteredFactories()}
				Learning rate: {self$learning_rate}
				Iterations: {self$getCurrentIteration()}
				")

      if (! is.null(self$positive_category))
        p = glue::glue(p, "\nPositive class: {self$positive_category}")

      if(! is.null(self$model))
        p = glue::glue(p, "\nOffset: {round(self$model$getOffset(), 4)}")

      print(p)
      print(self$loss)
    },
    getEstimatedCoef = function () {
      if(!is.null(self$model)) {
        return(c(self$model$getEstimatedParameter(), offset = self$model$getOffset()))
      }
      return(NULL)
    },
    plot = function (blearner_name = NULL, iters = NULL, from = NULL, to = NULL, length_out = 1000) {
      checkModelPlotAvailability(self)

      gg = plotFeatEffect(cboost_obj = self, bl_list = private$bl_list, blearner_name = blearner_name,
        iters = iters, from = from, to = to, length_out = length_out)

      return(gg)
    },
    getBaselearnerNames = function () {
      return(names(private$bl_list))
    },
    getLoggerData = function () {
      checkModelPlotAvailability(self, check_ggplot = FALSE)

      out_list = self$model$getLoggerData()
      out_mat = out_list[[2]]
      colnames(out_mat) = out_list[[1]]

      return(as.data.frame(out_mat[seq_len(self$getCurrentIteration()), , drop = FALSE]))
    },
    calculateFeatureImportance = function (num_feats = NULL) {
      checkModelPlotAvailability(self, check_ggplot = FALSE)

      max_feats = length(unique(self$getSelectedBaselearner()))
      checkmate::assert_integerish(x = num_feats, lower = 1, upper = max_feats, any.missing = FALSE, len = 1L, null.ok = TRUE)

      if (is.null(num_feats)) {
        num_feats = max_feats
        if (num_feats > 15L) { num_feats = 15L }
      }

      inbag_risk_differences = abs(diff(self$getInbagRisk()))
      selected_learner = self$getSelectedBaselearner()

      blearner_sums = aggregate(inbag_risk_differences, by = list(selected_learner), FUN = sum)
      colnames(blearner_sums) = c("baselearner", "relative_risk_reduction")
      blearner_sums[["relative_risk_reduction"]] = blearner_sums[["relative_risk_reduction"]] / sum(blearner_sums[["relative_risk_reduction"]])

      return(blearner_sums[order(blearner_sums[["relative_risk_reduction"]], decreasing = TRUE)[seq_len(num_feats)], ])
    },
    plotFeatureImportance = function (num_feats = NULL) {

      checkModelPlotAvailability(self)

      df_vip = self$calculateFeatureImportance(num_feats)

      gg = ggplot2::ggplot(df_vip, ggplot2::aes(x = reorder(baselearner, relative_risk_reduction), y = relative_risk_reduction)) +
        ggplot2::geom_bar(stat = "identity") + ggplot2::coord_flip() + ggplot2::ylab("Importance") + ggplot2::xlab("")

      return (gg)
    },
    plotInbagVsOobRisk = function () {

      checkModelPlotAvailability(self)

      inbag_trace = self$getInbagRisk()
      oob_data = self$getLoggerData()

      if ("oob_risk" %in% names(oob_data)) {
        oob_trace = oob_data[["oob_risk"]]

        df_risk = data.frame(
          risk = c(inbag_trace, oob_trace),
          type = rep(c("inbag", "oob"), times = c(length(inbag_trace), length(oob_trace))),
          iter = c(seq_along(inbag_trace), seq_along(oob_trace))
        )

        gg = ggplot2::ggplot(df_risk, ggplot2::aes(x = iter, y = risk, color = type)) +
          ggplot2::geom_line(size = 1.1) +
          ggplot2::xlab("Iteration") +
          ggplot2::ylab("Risk")# + labs(color = "")

        return(gg)
      } else {
        stop("Model was not trained with an out of bag risk logger called 'oob_risk'.")
      }
    }
  ),
  private = list(
    # Lists of single logger and base-learner factories. Necessary to prevent the factories from the
    # garbage collector which deallocates all the data from the heap and couses R to crash.
    l_list = list(),
    bl_list = list(),
    logger_list = list(),
    oob_idx = NULL,
    train_idx = NULL,
    initializeModel = function() {

      private$logger_list = LoggerList$new()
      lapply(private$l_list, function (logger) { private$logger_list$registerLogger(logger) })

      self$model = Compboost_internal$new(self$response, self$learning_rate,
        self$stop_if_all_stoppers_fulfilled, self$bl_factory_list, self$loss, private$logger_list, self$optimizer)
    },
    addOobLogger = function () {

      if (! is.null(self$oob_fraction)) {
        self$addLogger(logger = LoggerOobRisk, logger_id = "oob_risk",
          used.loss = self$loss, eps.for.break = 0, oob_data = self$prepareData(self$data_oob),
          oob.response = self$response_oob)
      }
    },
    addSingleNumericBl = function(data_columns, feature, id_fac, id, bl_factory, data_source, data_target, ...) {

      private$bl_list[[id]] = list()
      private$bl_list[[id]]$source = data_source$new(as.matrix(data_columns), paste(feature, collapse = "_"))
      private$bl_list[[id]]$feature = feature
      private$bl_list[[id]]$target = data_target$new()
      private$bl_list[[id]]$factory = bl_factory$new(private$bl_list[[id]]$source, private$bl_list[[id]]$target, id_fac, list(...))

      self$bl_factory_list$registerFactory(private$bl_list[[id]]$factory)
      private$bl_list[[id]]$source = NULL

    },
    addSingleCatBl = function(data_column, feature, id_fac, id, bl_factory, data_source, data_target, ...) {

      lvls = unlist(unique(data_column))

      # Create dummy variable for each category and use that vector as data matrix. Hence,
      # if a categorical feature has 3 groups, then these 3 groups are added as 3 different
      # base-learners (unbiased feature selection).
      for (lvl in lvls) {

        cat_feat_id = paste(feature, lvl, id_fac, sep = "_")

        private$addSingleNumericBl(data_columns = as.matrix(as.integer(data_column == lvl)),
          feature = paste(feature, lvl, sep = "_"), id_fac = id_fac,
          id = cat_feat_id, bl_factory, data_source, data_target, ...)

        # This is important because of:
        #   1. feature in addSingleNumericBl needs to be something like cat_feature_Group1 to define the
        #      data objects correctly in a unique way.
        #   2. The feature itself should not be named with the level. Instead of that we just want the
        #      feature name of the categorical variable, such as cat_feature (important for predictions).
        private$bl_list[[cat_feat_id]]$feature = feature
      }
    }
  )
)
