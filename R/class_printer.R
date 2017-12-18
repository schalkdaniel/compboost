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

# ---------------------------------------------------------------------------- #
# Baselearner:
# ---------------------------------------------------------------------------- #

setClass("Rcpp_BaselearnerWrapper")
setMethod("show", "Rcpp_BaselearnerWrapper", function (object) {
  cat("\nThis is a >>", object$GetBaselearnerType(), "<< baselearner:\n", sep = "")
  cat("\n  Formal S4 class:       ", class(object))
  cat("\n  Baselearner Identifier:", object$GetIdentifier())
  cat("\n\n")
})


# ---------------------------------------------------------------------------- #
# Loss:
# ---------------------------------------------------------------------------- #

setClass("Rcpp_LossWrapper")
setMethod("show", "Rcpp_LossWrapper", function (object) {
  cat("\nThis is a >>", object$GetLossName(), "<< loss:\n", sep = "")
  cat("\n  Formal S4 class:", class(object))
  cat("\n\n")
})

