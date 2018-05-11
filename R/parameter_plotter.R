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
#   R API which wraps the imported c++ class wrapper of the "Compboost" class
#   and acts as the accessor for the user to a high level function within R.
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

#' @title Parameter plotter for a trained compboost object
#'
#' @description This function can be used to print the trace of the parameters
#'   of a trained compboost object.
#'
#' @param object [\code{character(1)}] \cr
#'   Trained compboost object.
#' @param legend [\code{logical(1)}] \cr
#'   Logical to specify if a legend should be plotted.
#' @param ...
#'   Additional parameter given to plot.
#' @export

plotCompboostParameter = function (object, legend = TRUE, ...)
{
  if (! object$isTrained()) {
    warning ("Your given compboost object is not trained!")
    return (invisible(1))
  }
  
  plot.params = list(...)
  
  parameter.matrix = object$getParameterMatrix()
  
  parameter.matrix.df = as.data.frame(parameter.matrix$parameter.matrix)
  colnames(parameter.matrix.df) = parameter.matrix$parameter.names
  
  if (! "ylim" %in% names(plot.params)) {
    ylim = c(min(parameter.matrix.df), max(parameter.matrix.df))
  }
  if (! "xlab" %in% names(plot.params)) {
    xlab = "Iterations"
  }
  if (! "ylab" %in% names(plot.params)) {
    ylab = "Parameter Value"
  }
  if (! "type" %in% names(plot.params)) {
    type = "l"
  }
  if (! "col" %in% names(plot.params)) {
    col = rgb(
      red   = seq(0,   154, length.out = ncol(parameter.matrix.df)), 
      green = seq(178, 205, length.out = ncol(parameter.matrix.df)), 
      blue  = seq(238,  50, length.out = ncol(parameter.matrix.df)),
      alpha = 255, 
      maxColorValue = 255
    )
  }
  
  if (legend) {
    layout(mat =   matrix(
      data = c(
        2, 2, 2, 1, 1, 
        2, 2, 2, 1, 1, 
        2, 2, 2, 1, 1
      ),
      nrow  = 3,
      byrow = TRUE
    ))
    
    par(mar = c(0, 0, 0, 0))
    
    plot(1, type = "n", xlab = "", ylab = "", axes = FALSE, 
      xlim = c(0, 10), ylim = c(0, 10))
    
    legend(
      x = 0,
      y = 5.5,
      legend = colnames(parameter.matrix.df),
      lty = 1,
      lwd = 2,
      col = col,
      yjust = 0.5,
      xpd = TRUE,
      bty = "n"
    )
    par(mar = c(5.1, 4.1, 4.1, 2.1))
  }
  
  
  
  plot(
    x    = seq_len(nrow(parameter.matrix.df)),
    y    = parameter.matrix.df[, 1],
    ylim = ylim,
    xlab = xlab,
    ylab = ylab,
    type = type,
    col  = col[1],
    ...
  )
  
  if (ncol(parameter.matrix.df) > 1) {
    for (i in 2:ncol(parameter.matrix.df)) {
      points(
        x = seq_len(nrow(parameter.matrix.df)), 
        y = parameter.matrix.df[, i],
        type = "l",
        col = col[i]
      )
    }
  }
  par(mfrow = c(1,1))
  
  return (invisible(0))
}
