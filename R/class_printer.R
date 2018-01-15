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
#   The printer function for the 'BaselearnerWrapper'.
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

glueLoss = function (name, definition = NULL)
{
  if (is.null(definition)) {
    definition = "No function specified, probably you are using a custom loss."
  } else {
    definition = paste0("Loss function: y = ", definition)
  }
  
  ignore.me = glue::glue("

    {name} Loss:

      {definition}


    ")

  print(ignore.me)
  return(invisible(paste0(name, "Printer")))
}

# ---------------------------------------------------------------------------- #
# FacotryList:
# ---------------------------------------------------------------------------- #

setClass("Rcpp_FactoryList")
ignore.me = setMethod("show", "Rcpp_FactoryList", function (object) {
  cat("\n")
  object$printRegisteredFactorys()
  cat("\n\n")
  
  # For testing:
  return(invisible("FactoryListPrinter"))
})


# ---------------------------------------------------------------------------- #
# Loss:
# ---------------------------------------------------------------------------- #

setClass("Rcpp_QuadraticLoss")
ignore.me = setMethod("show", "Rcpp_QuadraticLoss", function (object) {
  glueLoss("Quadratic", "(y - f(x))^2")
})

setClass("Rcpp_AbsoluteLoss")
ignore.me = setMethod("show", "Rcpp_AbsoluteLoss", function (object) {
  glueLoss("Absolute", "|y - f(x)|")
})

setClass("Rcpp_CustomLoss")
ignore.me = setMethod("show", "Rcpp_CustomLoss", function (object) {
  glueLoss("Custom")
})

