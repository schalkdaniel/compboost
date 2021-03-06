% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{BaselearnerCategoricalBinary}
\alias{BaselearnerCategoricalBinary}
\title{Base-learner factory for categorical feature on a binary base-learner basis}
\format{
\code{\link{S4}} object.
}
\description{
\code{BaselearnerCategoricalBinary} can be used to estimate effects of one category of a categorical
feature. The base-learner gets the data as index vector of the observations assigned to the group.
For example, if the vector is (a, a, b, b, a, b), then the index vector is (1, 2, 5) for group a.
This reduces memory load and fasten the fitting process.
}
\section{Usage}{

\preformatted{
BaselearnerCategoricalBinary$new(data_source, list(n_obs))
}
}

\section{arguments}{

\describe{
\item{\code{data_source} [\code{data} object]}{
  data object of class \code{CategoricalData} which contains the source data.
}
}
}

\section{Fields}{

  This class doesn't contain public fields.
}

\section{Methods}{

\describe{
\item{\code{getData()}}{Get the data matrix of the target data which is used
  for modeling.}
\item{\code{transformData(X)}}{Transform a data matrix as defined within the
  factory. The argument has to be a matrix with one column. In case of the categorical
  binary base-learner this is the index of non-zero elements concatinated with the
  number of observations. This helps to fully reconstruct the original feature by using less memory. This also speed up computation time.}
\item{\code{summarizeFactory()}}{Summarize the base-learner factory object.}
}
}

\examples{
# Sample data:
x = sample(c("pos","neg"), 20, TRUE)

# Create new data object:
data_source = CategoricalData$new(x, "pos")

# Create new linear base-learner:
cat_factory = BaselearnerCategoricalBinary$new(data_source, "pos")

# Get the transformed data as stored for internal use:
cat_factory$getData()

# Summarize factory:
cat_factory$summarizeFactory()

}
