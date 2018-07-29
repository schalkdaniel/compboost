# ============================================================================ #
#                                  ___.                          __            #
#         ____  ____   _____ ______\_ |__   ____   ____  _______/  |_          #
#       _/ ___\/  _ \ /     \\____ \| __ \ /  _ \ /  _ \/  ___/\   __\         #
#       \  \__(  <_> )  Y Y  \  |_> > \_\ (  <_> |  <_> )___ \  |  |           #
#        \___  >____/|__|_|  /   __/|___  /\____/ \____/____  > |__|           #
#            \/            \/|__|       \/                  \/                 #
#                                                                              #
# ============================================================================ #
#
# Compboost is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# Compboost is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Compboost. If not, see <http:#www.gnu.org/licenses/>.
#
# This file contains:
# -------------------
#
#   The printer function for the exposed classes.
#
# Written by:
# -----------
#
#   Daniel Schalk
#   Institut für Statistik
#   Ludwig-Maximilians-Universität München
#   Ludwigstraße 33
#   D-80539 München
#
#   https:#www.compstat.statistik.uni-muenchen.de
#
# =========================================================================== #

# Helper functions:
# -----------------

glueLoss = function (name, definition = NULL, additional.desc = "")
{
  if (is.null(definition)) {
    definition = "No function specified, probably you are using a custom loss."
  } else {
    definition = paste0("Loss function: L(y,x) = ", definition)
  }

  ignore.me = glue::glue("

    {name} Loss:

      {definition}

      {additional.desc}

    ")

  print(ignore.me)
  return(invisible(paste0(name, "Printer")))
}

# ---------------------------------------------------------------------------- #
# Data:
# ---------------------------------------------------------------------------- #

setClass("Rcpp_InMemoryData")
ignore.me = setMethod("show", "Rcpp_InMemoryData", function (object) {

  cat("\n")

  if (object$getIdentifier() == "") {
    # data.type = "target"
    cat("Empty data object which can be used as target data.")
  } else {
    # data.type = "source"
    cat("Source Data: In memory class of feature ", object$getIdentifier(), ".")
    cat("\n             Note that using this class as target will override data specified before.")
  }

  cat("\n\n")

  return ("InMemoryDataPrinter")
})

# ---------------------------------------------------------------------------- #
# Factories:
# ---------------------------------------------------------------------------- #

setClass("Rcpp_BaselearnerPolynomial")
ignore.me = setMethod("show", "Rcpp_BaselearnerPolynomial", function (object) {
  cat("\n")
  object$summarizeFactory()
  cat("\n\n")

  return ("BaselearnerPolynomialPrinter")
})

setClass("Rcpp_BaselearnerPSpline")
ignore.me = setMethod("show", "Rcpp_BaselearnerPSpline", function (object) {
  cat("\n")
  object$summarizeFactory()
  cat("\n\n")

  return ("BaselearnerPSplinePrinter")
})

setClass("Rcpp_BaselearnerCustom")
ignore.me = setMethod("show", "Rcpp_BaselearnerCustom", function (object) {
  cat("\n")
  object$summarizeFactory()
  cat("\n\n")

  return ("BaselearnerCustomPrinter")
})

setClass("Rcpp_BaselearnerCustomCpp")
ignore.me = setMethod("show", "Rcpp_BaselearnerCustomCpp", function (object) {
  cat("\n")
  object$summarizeFactory()
  cat("\n\n")

  return ("BaselearnerCustomCppPrinter")
})

# ---------------------------------------------------------------------------- #
# BlearnerFacotryList:
# ---------------------------------------------------------------------------- #

setClass("Rcpp_BlearnerFactoryList")
ignore.me = setMethod("show", "Rcpp_BlearnerFactoryList", function (object) {
  cat("\n")
  object$printRegisteredFactories()
  cat("\n\n")

  # For testing:
  return(invisible("BlearnerFactoryListPrinter"))
})


# ---------------------------------------------------------------------------- #
# Loss:
# ---------------------------------------------------------------------------- #

setClass("Rcpp_LossQuadratic")
ignore.me = setMethod("show", "Rcpp_LossQuadratic", function (object) {
  glueLoss("LossQuadratic", "0.5 * (y - f(x))^2")
})

setClass("Rcpp_LossAbsolute")
ignore.me = setMethod("show", "Rcpp_LossAbsolute", function (object) {
  glueLoss("LossAbsolute", "|y - f(x)|")
})

setClass("Rcpp_LossBinomial")
ignore.me = setMethod("show", "Rcpp_LossBinomial", function (object) {
  # glueLoss("LossBinomial", "log(1 + exp(-2yf(x))", "Labels should be coded as -1 and 1!")
  glueLoss("LossBinomial", "log(1 + exp(-2yf(x))")
})

setClass("Rcpp_LossCustom")
ignore.me = setMethod("show", "Rcpp_LossCustom", function (object) {
  glueLoss("LossCustom")
})

setClass("Rcpp_LossCustomCpp")
ignore.me = setMethod("show", "Rcpp_LossCustomCpp", function (object) {
  glueLoss("LossCustomCpp")
})

# ---------------------------------------------------------------------------- #
# Logger:
# ---------------------------------------------------------------------------- #

setClass("Rcpp_LoggerIteration")
ignore.me = setMethod("show", "Rcpp_LoggerIteration", function (object) {
  cat("\n")
  object$summarizeLogger()
  cat("\n\n")

  return ("LoggerIterationPrinter")
})

setClass("Rcpp_LoggerInbagRisk")
ignore.me = setMethod("show", "Rcpp_LoggerInbagRisk", function (object) {
  cat("\n")
  object$summarizeLogger()
  cat("\n\n")

  return ("LoggerInbagRiskPrinter")
})

setClass("Rcpp_LoggerOobRisk")
ignore.me = setMethod("show", "Rcpp_LoggerOobRisk", function (object) {
  cat("\n")
  object$summarizeLogger()
  cat("\n\n")

  return ("LoggerOobRiskPrinter")
})

setClass("Rcpp_LoggerTime")
ignore.me = setMethod("show", "Rcpp_LoggerTime", function (object) {
  cat("\n")
  object$summarizeLogger()
  cat("\n\n")

  return ("LoggerTimePrinter")
})

# ---------------------------------------------------------------------------- #
# Loggerlist:
# ---------------------------------------------------------------------------- #

setClass("Rcpp_LoggerList")
ignore.me = setMethod("show", "Rcpp_LoggerList", function (object) {
  cat("\n")
  if (object$getNumberOfRegisteredLogger() == 0) {
    cat("No registered logger!")
  } else {
    object$printRegisteredLogger()
  }
  cat("\n\n")

  return ("LoggerListPrinter")
})

# ---------------------------------------------------------------------------- #
# Optimizer:
# ---------------------------------------------------------------------------- #

setClass("Rcpp_OptimizerCoordinateDescent")
ignore.me = setMethod("show", "Rcpp_OptimizerCoordinateDescent", function (object) {
  cat("\n")
  cat("Greedy optimizer! Optmizing over all baselearner in each iteration and",
    "choose the one with the lowest SSE.\n")
  cat("\n\n")

  return (invisible("OptimizerCoordinateDescentPrinter"))
})


# ---------------------------------------------------------------------------- #
# Compboost:
# ---------------------------------------------------------------------------- #

setClass("Rcpp_Compboost_internal")
ignore.me = setMethod("show", "Rcpp_Compboost_internal", function (object) {
  cat("\n")
  object$summarizeCompboost()
  cat("\n\n")

  return (invisible("CompboostInternalPrinter"))
})
