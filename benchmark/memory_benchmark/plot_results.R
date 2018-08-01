
library(ggplot2)
library(ggtech)
library(grid)
library(gridExtra)
library(gtable)

extrafont::loadfonts(device = "win")

plotMemResults = function (bm.algos.good, bm.algos.bad, mytitle, myxlab, ylim1, ylim2)
{
  gg.good = ggplot(bm.algos.good, aes(x = second, y = used.memory.centered, color = algo)) +
    geom_line(size = 1.5) +
    ggtitle("Linear Base-Learner") +
    ylab("Used Megabytes") +
    scale_color_tech(theme = "twitter") +
    ylim(ylim1, ylim2) +
    theme(
      panel.background = element_blank(),
      text             = element_text(family = "Palatino Linotype"),
      legend.title     = element_blank(),
      legend.key       = element_rect(fill = "transparent"),
      panel.grid.major = element_line(color = rgb(0.7, 0.7, 0.7, 0.4),
        size = 0.1, linetype = "dashed")
    )

  gg.bad = ggplot(bm.algos.bad, aes(x = second, y = used.memory.centered, color = algo)) +
    geom_line(size = 1.5) +
    ggtitle("Spline Base-Learner") +
    scale_color_tech(theme = "twitter") +
    ylim(ylim1, ylim2) +
    theme(
      panel.background = element_blank(),
      text             = element_text(family = "Palatino Linotype"),
      legend.title     = element_blank(),
      legend.key       = element_rect(fill = "transparent"),
      panel.grid.major = element_line(color = rgb(0.7, 0.7, 0.7, 0.4),
        size = 0.1, linetype = "dashed")
    )

  # Extracxt the legend from gg.iters.spline:
  legend = gtable_filter(ggplotGrob(gg.bad), "guide-box")

  # Make custom title:
  gtitle = textGrob(label = mytitle,
    vjust = 0.5, gp = gpar(fontfamily = "Palatino Linotype", fontface = "bold", cex = 1.5))

  # Arrange and draw the plot
  y.label = textGrob("", rot = 90, vjust = 0.5,
    gp = gpar(fontfamily = "Palatino Linotype"))

  x.label = textGrob(myxlab, vjust = -0.5,
    gp = gpar(fontfamily = "Palatino Linotype"))

  grid.arrange(y.label,
    arrangeGrob(
      gg.good + theme(legend.position="none") + xlab(""),
      gg.bad + theme(legend.position="none") + ylab("") + xlab(""),
      ncol = 2,
      top = gtitle,
      bottom = x.label
    ), legend,
    widths = unit.c(unit(2, "lines"), unit(1, "npc") - unit(2, "lines") - legend$width,
      legend$width), nrow=1)
}

# Baseline:
# -----------------------

source("mem_benchmark/figures/iters1234.R")

# 5000 Iterations:
# -----------------------

source("mem_benchmark/figures/iters5678.R")

# 2000 Base-Learner:
# -----------------------

source("mem_benchmark/figures/iters9101112.R")

# 50000 Observations:
# -----------------------

source("mem_benchmark/figures/iters13141516.R")
