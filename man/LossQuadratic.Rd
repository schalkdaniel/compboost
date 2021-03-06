% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{LossQuadratic}
\alias{LossQuadratic}
\title{Quadratic loss for regression tasks.}
\format{
\code{\link{S4}} object.
}
\description{
This loss can be used for regression with \eqn{y \in \mathrm{R}}.
}
\details{
\strong{Loss Function:}
\deqn{
  L(y, f(x)) = \frac{1}{2}( y - f(x))^2
}
\strong{Gradient:}
\deqn{
  \frac{\delta}{\delta f(x)}\ L(y, f(x)) = f(x) - y
}
\strong{Initialization:}
\deqn{
  \hat{f}^{[0]}(x) = \mathrm{arg~min}{c\in\mathrm{R}}{\mathrm{arg~min}}\ \frac{1}{n}\sum\limits_{i=1}^n
  L\left(y^{(i)}, c\right) = \bar{y}
}
}
\section{Usage}{

\preformatted{
LossQuadratic$new()
LossQuadratic$new(offset)
}
}

\section{Arguments}{

\describe{
\item{\code{offset} [\code{numeric(1)}]}{
  Numerical value which can be used to set a custom offset. If so, this
  value is returned instead of the loss optimal initialization.
}
}
}

\examples{

# Create new loss object:
quadratic_loss = LossQuadratic$new()
quadratic_loss

}
