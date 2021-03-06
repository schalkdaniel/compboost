% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{BlearnerFactoryList}
\alias{BlearnerFactoryList}
\title{Base-learner factory list to define the set of base-learners}
\format{
\code{\link{S4}} object.
}
\description{
\code{BlearnerFactoryList} creates an object in which base-learner factories
can be registered. This object can then be passed to compboost as set of
base-learner which is used by the optimizer to get the new best
base-learner.
}
\section{Usage}{

\preformatted{
BlearnerFactoryList$new()
}
}

\section{Fields}{

  This class doesn't contain public fields.
}

\section{Methods}{

\describe{
\item{\code{registerFactory(BaselearnerFactory)}}{Takes a object of the
  class \code{BaseLearnerFactory} and adds this factory to the set of
  base-learner.}
\item{\code{printRegisteredFactories()}}{Get all registered factories.}
\item{\code{clearRegisteredFactories()}}{Remove all registered factories.
  Note that the factories are not deleted, just removed from the map.}
\item{\code{getModelFrame()}}{Get each target data matrix parsed to one
  big matrix.}
\item{\code{getNumberOfRegisteredFactories()}}{Get the number of registered
  factories.}
}
}

\examples{
# Sample data:
data_mat = cbind(1:10)

# Create new data object:
data_source = InMemoryData$new(data_mat, "my_data_name")

lin_factory = BaselearnerPolynomial$new(data_source,
  list(degree = 1, intercept = TRUE))
poly_factory = BaselearnerPolynomial$new(data_source,
  list(degree = 2, intercept = TRUE))

# Create new base-learner list:
my_bl_list = BlearnerFactoryList$new()

# Register factories:
my_bl_list$registerFactory(lin_factory)
my_bl_list$registerFactory(poly_factory)

# Get registered factories:
my_bl_list$printRegisteredFactories()

# Get all target data matrices in one big matrix:
my_bl_list$getModelFrame()

# Clear list:
my_bl_list$clearRegisteredFactories()

# Get number of registered factories:
my_bl_list$getNumberOfRegisteredFactories()

}
