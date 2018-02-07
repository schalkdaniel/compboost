#' # ============================================================================ #
#' #                                  ___.                          __            #
#' #         ____  ____   _____ ______\_ |__   ____   ____  _______/  |_          #
#' #       _/ ___\/  _ \ /     \\____ \| __ \ /  _ \ /  _ \/  ___/\   __\         #
#' #       \  \__(  <_> )  Y Y  \  |_> > \_\ (  <_> |  <_> )___ \  |  |           #
#' #        \___  >____/|__|_|  /   __/|___  /\____/ \____/____  > |__|           #
#' #            \/            \/|__|       \/                  \/                 #
#' #                                                                              #
#' # ============================================================================ #
#' #
#' # Compboost is free software: you can redistribute it and/or modify
#' # it under the terms of the GNU General Public License as published by
#' # the Free Software Foundation, either version 3 of the License, or
#' # (at your option) any later version.
#' # Compboost is distributed in the hope that it will be useful,
#' # but WITHOUT ANY WARRANTY; without even the implied warranty of
#' # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#' # GNU General Public License for more details.
#' # You should have received a copy of the GNU General Public License
#' # along with Compboost. If not, see <http:#www.gnu.org/licenses/>.
#' #
#' # This file contains:
#' # -------------------
#' #
#' #   R API which wraps the imported c++ class wrapper of the "Compboost" class
#' #   and acts as the accessor for the user to a high level function within R.
#' #
#' # Written by:
#' # -----------
#' #
#' #   Daniel Schalk
#' #   Institut für Statistik
#' #   Ludwig-Maximilians-Universität München
#' #   Ludwigstraße 33
#' #   D-80539 München
#' #
#' #   https:#www.compstat.statistik.uni-muenchen.de
#' #
#' # =========================================================================== #
#' 
#' #' @title Compboost main R Function
#' #'
#' #' @description This function provides an R API to create the compboost class 
#' #'   object automatically. Notice that it is also possible to initialize 
#' #'   everything by hand'.
#' #'
#' #' @param name [\code{character(1)}] \cr
#' #'   Name to initialize the C++Class object.
#' #' @return [\code{C++Class}] \cr
#' #'   A object of class 'CompboostWrapper'.
#' #' @export
#' NULL
