# =================================================================================================== #
#                                                                                                     #
#                             Function to Create Plots Out of The Benchmark                           #
#                                                                                                     #
# =================================================================================================== #

#' Barplots to compare the runtime between compboost, mboost, and gamboost/glmboost
#' 
#' This function takes a data.frame/data.table and create a dodged barplot of the 
#' runtime. Each is done for linear and spline base-learner. Additionally, there are
#' also barplots that illustrates the relative difference of mboost and gamboost/glmboost
#' compared to compboost.
#' 
#' @param data [\code{data.frame}]\cr
#'   The data object containing the benchmark results for a specific task (e.g. increasing
#'   number of iterations). The data object must have a special structure. Especially the 
#'   column names must exactly names as followed:
#'     - learner: Factor variable containing if the base-learners are linear or spline.
#'     - x.value: This value contains the task (e.g. increasing iterations, an increasing dimension, 
#'                and so on).
#'     - Time: An aggregated time value like the median of different replications.
#'     - Algorithm: A factor variable indicating the used algorithm.
#'     - Time.min: Another time value containing a lower boundary of the elapsed times (e.g. the
#'                 minimum). This value is used for the error bars.
#'     - Time.max: Same as Time.min but now applies to the upper boundary.
#'     - rel.factor: The relative factor of the single algorithm compared to a base-line.
#' @param header [\code{character(1)}]\cr
#'   Header of the graphic passed as string.
#' @param xlab [\code{character(1)}]\cr
#'   Label of the x axis passed as string.
#' @example  
#' mydf = data.frame(
#'   learner = ,
#'   x.value = ,
#'   Time    = ,
#'   Algorithm = ,
#'   Time.min  = ,
#'   Time.max  = 
#' )
#' plotRuntimeBenchmark(mydf, "My Benchmark Graphic", "This is just a test")
plotRuntimeBenchmark = function (data, header, xlab) {

  ylim.upper = max(data$Time.max)

  plotAbsoluteTime = function (data, sel.learner) 
  {
    data %>%
      filter(learner == sel.learner) %>%
      ggplot(aes(x = reorder(x.value, x.value), y = Time, fill = Algorithm,
        ymin = Time.min, ymax = Time.max)) +
      geom_col(position = "dodge", width = 0.7) +
      geom_errorbar(width = 0.2, position = position_dodge(0.7), colour = rgb(1, 0.3, 0.2)) +
      scale_fill_manual(values = mycolors) +
      ggtitle("Linear Base-Learner") +
      ylab("Elapsed Time\nin Minutes") +
      theme(
        panel.background = element_blank(),
        legend.position  = "none",
        panel.grid.major = element_line(color = rgb(0.7, 0.7, 0.7, 0.4),
          size = 0.1, linetype = "dashed")
      )
  }

  plotRelativeTime = function (data, sel.learner)
  {
    data %>% 
      filter(learner == sel.learner) %>%
      ggplot(aes(x = reorder(x.value, x.value), y = rel.factor, fill = Algorithm)) +
      geom_col(position = "dodge", width = 0.2) +
      scale_fill_manual(values = mycolors) +
      ggtitle("") +
      ylab("Relative\nRuntime") +
      theme(
        panel.background = element_blank(),
        legend.position  = "none",
        panel.grid.major = element_line(color = rgb(0.7, 0.7, 0.7, 0.4),
          size = 0.1, linetype = "dashed")
      )
  }

	layout.mat = matrix(
    data = c(
      1, 1, 2, 2,
      1, 1, 2, 2,
      1, 1, 2, 2,
      3, 3, 4, 4
    ), nrow = 4, byrow = TRUE
  )

  # Get colors from twitter colors:
  mycolors = c(compboost = "#55ACEE", mboost = "#292f33", gamboost = "#8899a6", glmboost = "#8c4183")

  # Get dummy barplot to extract the full legend:
  gg.legend = data %>%
    ggplot(aes(Algorithm, fill = Algorithm)) + geom_bar() + scale_fill_manual(values = mycolors)

	# Plot linear learner:
  gg.linear = data %>% plotAbsoluteTime("linear")
  gg.linear.rel = data %>% plotRelativeTime("linear")

  # Plot spline learner:
  gg.spline = data %>% plotAbsoluteTime("spline")
  gg.spline.rel = data %>% plotRelativeTime("spline")
  
  # Extracxt the legend from the dummy plot gg.legend:
  legend = gtable_filter(ggplotGrob(gg.legend), "guide-box")
  
  # Make custom title:
  gtitle = textGrob(label = header, vjust = 0.5, gp = gpar(fontface = "bold", cex = 1.5))
  
  # Arrange and draw the plot
  y.label = textGrob("", rot = 90, vjust = 0.5)  
  x.label = textGrob(xlab, vjust = -0.5)

  
  grid.arrange(y.label,
    arrangeGrob(
      gg.linear + theme(legend.position="none") + xlab("") + ylim(0, ylim.upper * 1.1),
      gg.spline + theme(legend.position="none") + ylab("") + xlab("") + ylim(0, ylim.upper * 1.1),
      gg.linear.rel + xlab(""),
      gg.spline.rel + xlab(""),
      layout_matrix = layout.mat,
      top = gtitle,
      bottom = x.label
    ), legend,
    widths = unit.c(unit(2, "lines"), unit(1, "npc") - unit(2, "lines") - legend$width,
      legend$width), nrow=1)
}