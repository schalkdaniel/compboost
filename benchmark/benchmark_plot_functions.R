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
#' time = runif(42, 5, 100)
#' mydf = data.frame(
#'   learner = rep(c("linear", "spline"), each = 21L),
#'   x.value = as.factor(rep(c(100, 500, 1000, 2000, 5000, 10000, 15000), each = 3L)),
#'   Time    = time,
#'   Algorithm = c(rep(c("compboost", "glmboost", "mboost"), times = 7L), rep(c("compboost", "gamboost", "mboost"), times = 7L)),
#'   Time.min  = time * 0.9,
#'   Time.max  = time * 1.1,
#'   rel.factor = runif(42, 0, 10), 
#'   stringsAsFactors = TRUE
#' )
#' plotRuntimeBenchmark(mydf, "My Benchmark Graphic", "This is just a test")
plotRuntimeBenchmark = function (data, header, xlab) 
{
  # Define colors:
  colors.twitter = list(
    main = c(compboost = "#55ACEE", mboost = "#292f33", gamboost = "#8899a6", glmboost = "#8c4183"),
    errorbar = "#cf2315"
  )
  colors.kandinsky = list(
    main = c(compboost = "#ce675e", mboost = "#98c4cf", gamboost = "#c7ad3c", glmboost = "#8bb09e"),
    errorbar = "#1a1a1c"
  )

  # mycolors = colors.kandinsky
  mycolors = colors.twitter

  plotAbsoluteTime = function (data, sel.learner) 
  {
    if (sel.learner == "linear") {
      gg.ll = "Linear"
    } else {
      gg.ll = "Spline"
    }
    data %>%
      filter(learner == sel.learner) %>%
      ggplot(aes(x = reorder(x.value, x.value), y = Time, fill = Algorithm, ymin = Time.min, ymax = Time.max)) +
      geom_col(position = "dodge", width = 0.7) +
      geom_errorbar(width = 0.2, position = position_dodge(0.7), colour = mycolors[["errorbar"]]) +
      scale_fill_manual(values = mycolors[["main"]]) +
      ggtitle(paste0(gg.ll, " Base-Learner")) +
      ylab("Elapsed Time\nin Minutes") +
      theme(
        panel.spacing = unit(0, "lines"),
        panel.background = element_rect(fill = "transparent"),
        plot.background = element_rect(fill = "transparent"),
        legend.background = element_rect(fill = "transparent", size=0, colour = NA),
        legend.box.background = element_rect(fill = "transparent", colour = "transparent"),
        panel.grid.minor = element_blank(),
        text             = element_text(),
        legend.position  = "none",
        panel.grid.major = element_line(
          color = rgb(0.7, 0.7, 0.7, 0.4),
          size  = 0.1, 
          linetype = "dashed"
        )
      )
  }

  plotRelativeTime = function (data, sel.learner)
  {
    data %>% 
      filter(learner == sel.learner) %>%
      ggplot(aes(x = reorder(x.value, x.value), y = rel.factor, fill = Algorithm)) +
      geom_col(position = "dodge", width = 0.2) +
      scale_fill_manual(values = mycolors[["main"]]) +
      ggtitle("") +
      ylab("Relative\nRuntime") +
      theme(
        panel.spacing = unit(0, "lines"),
        panel.background = element_rect(fill = "transparent"),
        plot.background  = element_rect(fill = "transparent"),
        legend.position  = "none",
        panel.grid.minor = element_blank(),
        text             = element_text(),
        legend.title     = element_blank(),
        panel.grid.major = element_line(
          color = rgb(0.7, 0.7, 0.7, 0.4),
          size  = 0.1, 
          linetype = "dashed"
        )
      )
  }

	layout.mat = matrix(
    data = c(
      1, 1, 2, 2,
      1, 1, 2, 2,
      1, 1, 2, 2,
      3, 3, 4, 4
    ), 
    ncol = 4, byrow = TRUE
  )

  # Get dummy barplot to extract the full legend:
  gg.legend = data %>%
    ggplot(aes(Algorithm, fill = Algorithm)) + geom_bar() + scale_fill_manual(values = mycolors[["main"]], guide = guide_legend(nrow = 1)) +
      theme(legend.title = element_blank(), legend.text = element_text(margin = margin(r = 24, l = 4, unit = "pt"))) 

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

  return (
    grid.arrange(y.label,
      arrangeGrob(
        arrangeGrob(
          gg.linear + theme(legend.position="none") + xlab(""), # + ylim(0, ylim.upper * 1.1),
          gg.spline + theme(legend.position="none") + xlab("") + ylab(""), # + ylim(0, ylim.upper * 1.1),
          gg.linear.rel + xlab(""),
          gg.spline.rel + xlab("") + ylab(""),
          layout_matrix = layout.mat,
          top = gtitle,
          bottom = x.label
        ),
        legend,
        layout_matrix = matrix(data = c(rep(1, 25), rep(2, 5)), ncol = 5, byrow = TRUE) 
      ),
      widths = unit.c(unit(2, "lines"), unit(1, "npc") - unit(2, "lines"),
        unit(0, "lines"))
    )
  )
}