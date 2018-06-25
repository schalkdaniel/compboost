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
#   The function to load the Rcpp module whithin R.
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

Rcpp::loadModule(module = "compboost_module", what = TRUE)
Rcpp::loadModule(module = "loss_module", what = TRUE)
Rcpp::loadModule(module = "baselearner_factory_module", what = TRUE)
Rcpp::loadModule(module = "baselearner_list_module", what = TRUE)
Rcpp::loadModule(module = "logger_module", what = TRUE)
Rcpp::loadModule(module = "optimizer_module", what = TRUE)
Rcpp::loadModule(module = "data_module", what = TRUE)

