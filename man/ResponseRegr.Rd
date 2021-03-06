% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{ResponseRegr}
\alias{ResponseRegr}
\title{Create response object for regression.}
\format{
\code{\link{S4}} object.
}
\description{
\code{ResponseRegr} creates a response object that are used as target during the
fitting process.
}
\section{Usage}{

\preformatted{
ResponseRegr$new(target_name, response)
ResponseRegr$new(target_name, response, weights)
}
}

\examples{

response_regr = ResponseRegr$new("target", cbind(rnorm(10)))
response_regr$getResponse()
response_regr$getTargetName()

}
