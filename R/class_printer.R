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
    definition = paste0("Loss function: y = ", definition)
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
# Baselearner:
# ---------------------------------------------------------------------------- #

setClass("Rcpp_PolynomialBlearner")
ignore.me = setMethod("show", "Rcpp_PolynomialBlearner", function (object) {
  cat("\n")
  object$summarizeBaselearner()
  cat("\n\n")
  
  return ("PolynomialBlearnerPrinter")
})

setClass("Rcpp_PSplineBlearner")
ignore.me = setMethod("show", "Rcpp_PSplineBlearner", function (object) {
  cat("\n")
  object$summarizeBaselearner()
  cat("\n\n")
  
  return ("PSplineBlearnerPrinter")
})

setClass("Rcpp_CustomBlearner")
ignore.me = setMethod("show", "Rcpp_CustomBlearner", function (object) {
  cat("\n")
  object$summarizeBaselearner()
  cat("\n\n")
  
  return ("CustomBlearnerPrinter")
})

setClass("Rcpp_CustomCppBlearner")
ignore.me = setMethod("show", "Rcpp_CustomCppBlearner", function (object) {
  cat("\n")
  object$summarizeBaselearner()
  cat("\n\n")
  
  return ("CustomCppBlearnerPrinter")
})

# ---------------------------------------------------------------------------- #
# Factories:
# ---------------------------------------------------------------------------- #

setClass("Rcpp_PolynomialBlearnerFactory")
ignore.me = setMethod("show", "Rcpp_PolynomialBlearnerFactory", function (object) {
  cat("\n")
  object$summarizeFactory()
  cat("\n\n")
  
  return ("PolynomialBlearnerFactoryPrinter")
})

setClass("Rcpp_PSplineBlearnerFactory")
ignore.me = setMethod("show", "Rcpp_PSplineBlearnerFactory", function (object) {
  cat("\n")
  object$summarizeFactory()
  cat("\n\n")
  
  return ("PSplineBlearnerFactoryPrinter")
})

setClass("Rcpp_CustomBlearnerFactory")
ignore.me = setMethod("show", "Rcpp_CustomBlearnerFactory", function (object) {
  cat("\n")
  object$summarizeFactory()
  cat("\n\n")
  
  return ("CustomBlearnerFactoryPrinter")
})

setClass("Rcpp_CustomCppBlearnerFactory")
ignore.me = setMethod("show", "Rcpp_CustomCppBlearnerFactory", function (object) {
  cat("\n")
  object$summarizeFactory()
  cat("\n\n")
  
  return ("CustomCppBlearnerFactoryPrinter")
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

setClass("Rcpp_QuadraticLoss")
ignore.me = setMethod("show", "Rcpp_QuadraticLoss", function (object) {
  glueLoss("QuadraticLoss", "(y - f(x))^2")
})

setClass("Rcpp_AbsoluteLoss")
ignore.me = setMethod("show", "Rcpp_AbsoluteLoss", function (object) {
  glueLoss("AbsoluteLoss", "|y - f(x)|")
})

setClass("Rcpp_BinomialLoss")
ignore.me = setMethod("show", "Rcpp_BinomialLoss", function (object) {
  glueLoss("BinomialLoss", "log(1 + exp(-2yf(x))", "Labels should be coded as -1 and 1!")
})

setClass("Rcpp_CustomLoss")
ignore.me = setMethod("show", "Rcpp_CustomLoss", function (object) {
  glueLoss("CustomLoss")
})

setClass("Rcpp_CustomCppLoss")
ignore.me = setMethod("show", "Rcpp_CustomCppLoss", function (object) {
  glueLoss("CustomCppLoss")
})

# ---------------------------------------------------------------------------- #
# Logger:
# ---------------------------------------------------------------------------- #

setClass("Rcpp_IterationLogger")
ignore.me = setMethod("show", "Rcpp_IterationLogger", function (object) {
  cat("\n")
  object$summarizeLogger()
  cat("\n\n")
  
  return ("IterationLoggerPrinter")
})

setClass("Rcpp_InbagRiskLogger")
ignore.me = setMethod("show", "Rcpp_InbagRiskLogger", function (object) {
  cat("\n")
  object$summarizeLogger()
  cat("\n\n")
  
  return ("InbagRiskLoggerPrinter")
})

setClass("Rcpp_OobRiskLogger")
ignore.me = setMethod("show", "Rcpp_OobRiskLogger", function (object) {
  cat("\n")
  object$summarizeLogger()
  cat("\n\n")
  
  return ("OobRiskLoggerPrinter")
})

setClass("Rcpp_TimeLogger")
ignore.me = setMethod("show", "Rcpp_TimeLogger", function (object) {
  cat("\n")
  object$summarizeLogger()
  cat("\n\n")
  
  return ("TimeLoggerPrinter")
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

setClass("Rcpp_GreedyOptimizer")
ignore.me = setMethod("show", "Rcpp_GreedyOptimizer", function (object) {
  cat("\n")
  cat("Greedy optimizer! Optmizing over all baselearner in each iteration and",
    "choose the one with the lowest SSE.\n")
  cat("\n\n")
  
  return (invisible("GreedyOptimizerPrinter"))
})


# ---------------------------------------------------------------------------- #
# Compboost:
# ---------------------------------------------------------------------------- #

setClass("Rcpp_Compboost")
ignore.me = setMethod("show", "Rcpp_Compboost", function (object) {
  cat("\n")
  object$summarizeCompboost()
  cat("\n\n")
  
  return (invisible("CompboostPrinter"))
})
