# ============================================================================ #
#                                                                              #
#                    Visualizing Results of Benchmark                          #
#                                                                              #
# ============================================================================ #

library(dplyr)
library(tidyr)
library(ggplot2)
library(ggtech)
library(grid)
library(gridExtra)
library(gtable)
library(batchtools)

# Load required fonts:
if (Sys.info()["sysname"] == "Windows") {
  extrafont::loadfonts(device = "win")
}

layout.mat = matrix(
  data = c(
    1, 1, 2, 2,
    1, 1, 2, 2,
    1, 1, 2, 2,
    3, 3, 4, 4
  ), nrow = 4, byrow = TRUE
)

# Suppress scientific format:
options(scipen=10000)

# Get results:
# -----------------------------------------

regis = loadRegistry("runtime_benchmark/cboost_bm")
regis$writeable = TRUE

res.list = unwrap(reduceResultsDataTable())

# Time in Minutes:
res.list$time = sapply(res.list$time, function(x) as.numeric(x)[3]) / 60
res.list$nrows = sapply(res.list$data.dim, function(x) as.numeric(x)[1])
res.list$ncols = sapply(res.list$data.dim, function(x) as.numeric(x)[2])


# Plot for Increasing Iterations:
# -----------------------------------------

# Transform Data:
dt.iterations = res.list %>%
  filter(job.id <= 140) %>%
  group_by(learner, iters, algo) %>%
  summarize(Iterations = iters[1], Time = median(time), Algorithm = algo[1],
    Time.min = min(time), Time.max = max(time)) %>%
  mutate(rel.factor = Time[algo == "mboost"] / Time[algo == "compboost"])

# Plot linear learner:
gg.iters.linear = dt.iterations %>%
  filter(learner == "linear") %>%
  ggplot(aes(x = reorder(Iterations, Iterations), y = Time, fill = Algorithm,
    ymin = Time.min, ymax = Time.max)) +
  geom_col(position = "dodge", width = 0.7) +
  geom_errorbar(width = 0.2, position = position_dodge(0.7), colour = rgb(1, 0.3, 0.2)) +
  scale_fill_tech(theme = "twitter") +
  ggtitle("Linear Base-Learner") +
  ylab("Elapsed Time\nin Minuts") +
  theme(
    panel.background = element_blank(),
    text             = element_text(family = "Palatino Linotype"),
    legend.position  = "none",
    panel.grid.major = element_line(color = rgb(0.7, 0.7, 0.7, 0.4),
      size = 0.1, linetype = "dashed")
  )

gg.iters.linear.rel = dt.iterations %>%
  filter(learner == "linear" & Algorithm == "compboost") %>%
  ggplot(aes(x = reorder(Iterations, Iterations), y = rel.factor)) +
  geom_col(position = "dodge", width = 0.1, fill = "#55acee") +
  scale_fill_tech(theme = "twitter") +
  ggtitle("") +
  ylab("Relative\nRuntime") +
  theme(
    panel.background = element_blank(),
    text             = element_text(family = "Palatino Linotype"),
    legend.position  = "none",
    panel.grid.major = element_line(color = rgb(0.7, 0.7, 0.7, 0.4),
      size = 0.1, linetype = "dashed")
  )

# Plot spline learner:
gg.iters.spline = dt.iterations %>%
  filter(learner == "spline") %>%
  ggplot(aes(x = reorder(Iterations, Iterations), y = Time, fill = Algorithm,
    ymin = Time.min, ymax = Time.max)) +
  geom_col(position = "dodge", width = 0.7) +
  geom_errorbar(width = 0.2, position = position_dodge(0.7), colour = rgb(1, 0.3, 0.2)) +
  scale_fill_tech(theme = "twitter") +
  ggtitle("P-Spline Base-Learner") +
  theme(
    panel.background = element_blank(),
    text             = element_text(family = "Palatino Linotype"),
    legend.title     = element_blank(),
    panel.grid.major = element_line(color = rgb(0.7, 0.7, 0.7, 0.4),
      size = 0.1, linetype = "dashed")
  )


gg.iters.spline.rel = dt.iterations %>%
  filter(learner == "spline" & Algorithm == "compboost") %>%
  ggplot(aes(x = reorder(Iterations, Iterations), y = rel.factor)) +
  geom_col(position = "dodge", width = 0.1, fill = "#55acee") +
  scale_fill_tech(theme = "twitter") +
  ggtitle("") +
  ylab("") +
  theme(
    panel.background = element_blank(),
    text             = element_text(family = "Palatino Linotype"),
    legend.position  = "none",
    panel.grid.major = element_line(color = rgb(0.7, 0.7, 0.7, 0.4),
      size = 0.1, linetype = "dashed")
  )

# Extracxt the legend from gg.iters.spline:
legend = gtable_filter(ggplotGrob(gg.iters.spline), "guide-box")

# Make custom title:
gtitle = textGrob(label = "Benchmark for Increasing Number of Iterations",
  vjust = 0.5, gp = gpar(fontfamily = "Palatino Linotype", fontface = "bold", cex = 1.5))

# Arrange and draw the plot
y.label = textGrob("", rot = 90, vjust = 0.5,
  gp = gpar(fontfamily = "Palatino Linotype"))

x.label = textGrob("Number of Iterations", vjust = -0.5,
  gp = gpar(fontfamily = "Palatino Linotype"))

grid.arrange(y.label,
  arrangeGrob(
    gg.iters.linear + theme(legend.position="none") + xlab("") + ylim(0, 100),
    gg.iters.spline + theme(legend.position="none") + ylab("") + xlab("") + ylim(0, 100),
    gg.iters.linear.rel + xlab("") + ylim(0, 11),
    gg.iters.spline.rel + xlab("") + ylim(0, 11),
    layout_matrix = layout.mat,
    top = gtitle,
    bottom = x.label
  ), legend,
  widths = unit.c(unit(2, "lines"), unit(1, "npc") - unit(2, "lines") - legend$width,
    legend$width), nrow=1)




# Plot for Increasing number of learner:
# -----------------------------------------

# Transform Data:
dt.learner = res.list %>%
  filter((job.id > 140 & job.id <= 210) | (job.id > 280 & job.id <= 350)) %>%
  group_by(learner, algo, ncols) %>%
  summarize(nblearner = ncols[1] - 1, Time = median(time), Algorithm = algo[1],
    Time.min = min(time), Time.max = max(time))

if (! 4000 %in% dt.learner$nblearner[dt.learner$algo == "mboost"]) {
  dt.learner = dt.learner %>%
    as.data.frame() %>%
    add_row(learner = c("linear", "spline"), algo = "mboost", ncols = 4001,
      nblearner = 4000, Time = NA, Algorithm = "mboost", Time.min = NA, Time.max = NA)
}

dt.learner = dt.learner %>%
  arrange(learner, nblearner)

dt.learner = dt.learner %>%
  as.data.table() %>%
  mutate(rel.factor = rep(Time[algo == "mboost"], each = 2) / Time)


# Plot linear learner:
gg.learner.linear = dt.learner %>%
  filter(learner == "linear") %>%
  ggplot(aes(x = reorder(nblearner, nblearner), y = Time, fill = Algorithm,
    ymin = Time.min, ymax = Time.max)) +
  geom_col(position = "dodge", width = 0.7) +
  geom_errorbar(width = 0.2, position = position_dodge(0.7), colour = rgb(1, 0.3, 0.2)) +
  scale_fill_tech(theme = "twitter") +
  ggtitle("Linear Base-Learner") +
  ylab("Elapsed Time\nin Minuts") +
  theme(
    panel.background = element_blank(),
    text             = element_text(family = "Palatino Linotype"),
    legend.position  = "none",
    panel.grid.major = element_line(color = rgb(0.7, 0.7, 0.7, 0.4),
      size = 0.1, linetype = "dashed")
  )

gg.learner.linear.rel = dt.learner %>%
  filter(learner == "linear" & Algorithm == "compboost") %>%
  ggplot(aes(x = reorder(nblearner, nblearner), y = rel.factor)) +
  geom_col(position = "dodge", width = 0.1, fill = "#55acee") +
  scale_fill_tech(theme = "twitter") +
  ggtitle("") +
  ylab("Relative\nRuntime") +
  theme(
    panel.background = element_blank(),
    text             = element_text(family = "Palatino Linotype"),
    legend.position  = "none",
    panel.grid.major = element_line(color = rgb(0.7, 0.7, 0.7, 0.4),
      size = 0.1, linetype = "dashed")
  )

# Plot spline learner:
gg.learner.spline = dt.learner %>%
  filter(learner == "spline") %>%
  ggplot(aes(x = reorder(nblearner, nblearner), y = Time, fill = Algorithm,
    ymin = Time.min, ymax = Time.max)) +
  geom_col(position = "dodge", width = 0.7) +
  geom_errorbar(width = 0.2, position = position_dodge(0.7), colour = rgb(1, 0.3, 0.2)) +
  scale_fill_tech(theme = "twitter") +
  ggtitle("P-Spline Base-Learner") +
  theme(
    panel.background = element_blank(),
    text             = element_text(family = "Palatino Linotype"),
    legend.title     = element_blank(),
    panel.grid.major = element_line(color = rgb(0.7, 0.7, 0.7, 0.4),
      size = 0.1, linetype = "dashed")
  )

gg.learner.spline.rel = dt.learner %>%
  filter(learner == "spline" & Algorithm == "compboost") %>%
  ggplot(aes(x = reorder(nblearner, nblearner), y = rel.factor)) +
  geom_col(position = "dodge", width = 0.1, fill = "#55acee") +
  scale_fill_tech(theme = "twitter") +
  ggtitle("") +
  ylab("") +
  theme(
    panel.background = element_blank(),
    text             = element_text(family = "Palatino Linotype"),
    legend.position  = "none",
    panel.grid.major = element_line(color = rgb(0.7, 0.7, 0.7, 0.4),
      size = 0.1, linetype = "dashed")
  )


# Extracxt the legend from gg.iters.spline:
legend = gtable_filter(ggplotGrob(gg.learner.spline), "guide-box")

# Make custom title:
gtitle = textGrob(label = "Benchmark for Increasing Number of Base-Learner",
  vjust = 0.5, gp = gpar(fontfamily = "Palatino Linotype", fontface = "bold", cex = 1.5))

# Arrange and draw the plot
y.label = textGrob("", rot = 90, vjust = 0.5,
  gp = gpar(fontfamily = "Palatino Linotype"))

x.label = textGrob("Number of Base-Learner", vjust = -0.5,
  gp = gpar(fontfamily = "Palatino Linotype"))

grid.arrange(y.label,
  arrangeGrob(
    gg.learner.linear + theme(legend.position="none") + xlab("") + ylim(0, 20),
    gg.learner.spline + theme(legend.position="none") + ylab("") + xlab("") + ylim(0, 20),
    gg.learner.linear.rel + xlab("") + ylim(0, 7),
    gg.learner.spline.rel + xlab("") + ylim(0, 7),
    layout_matrix = layout.mat,
    top = gtitle,
    bottom = x.label
  ), legend,
  widths = unit.c(unit(2, "lines"), unit(1, "npc") - unit(2, "lines") - legend$width,
    legend$width), nrow=1)


# Plot for Increasing Rows:
# -----------------------------------------

# Transform Data:
dt.rows = res.list %>%
  filter((job.id > 210 & job.id <= 280) | (job.id >350)) %>%
  group_by(learner, nrows, algo) %>%
  summarize(n.rows = nrows[1], Time = median(time), Algorithm = algo[1],
    Time.min = min(time), Time.max = max(time)) %>%
  mutate(rel.factor = Time[algo == "mboost"] / Time[algo == "compboost"])

# Plot linear learner:
gg.rows.linear = dt.rows %>%
  filter(learner == "linear") %>%
  ggplot(aes(x = reorder(n.rows, n.rows), y = Time, fill = Algorithm,
    ymin = Time.min, ymax = Time.max)) +
  geom_col(position = "dodge", width = 0.7) +
  geom_errorbar(width = 0.2, position = position_dodge(0.7), colour = rgb(1, 0.3, 0.2)) +
  scale_fill_tech(theme = "twitter") +
  ggtitle("Linear Base-Learner") +
  ylab("Elapsed Time\nin Minutes") +
  theme(
    panel.background = element_blank(),
    text             = element_text(family = "Palatino Linotype"),
    legend.position  = "none",
    panel.grid.major = element_line(color = rgb(0.7, 0.7, 0.7, 0.4),
      size = 0.1, linetype = "dashed")
  )

gg.rows.linear.rel = dt.rows %>%
  filter(learner == "linear" & Algorithm == "compboost") %>%
  ggplot(aes(x = reorder(n.rows, n.rows), y = rel.factor)) +
  geom_col(position = "dodge", width = 0.1, fill = "#55acee") +
  scale_fill_tech(theme = "twitter") +
  ggtitle("") +
  ylab("Relative\nRuntime") +
  theme(
    panel.background = element_blank(),
    text             = element_text(family = "Palatino Linotype"),
    legend.position  = "none",
    panel.grid.major = element_line(color = rgb(0.7, 0.7, 0.7, 0.4),
      size = 0.1, linetype = "dashed")
  )


# Plot spline learner:
gg.rows.spline = dt.rows %>%
  filter(learner == "spline") %>%
  ggplot(aes(x = reorder(n.rows, n.rows), y = Time, fill = Algorithm,
    ymin = Time.min, ymax = Time.max)) +
  geom_col(position = "dodge", width = 0.7) +
  geom_errorbar(width = 0.2, position = position_dodge(0.7), colour = rgb(1, 0.3, 0.2)) +
  scale_fill_tech(theme = "twitter") +
  ggtitle("P-Spline Base-Learner") +
  theme(
    panel.background = element_blank(),
    text             = element_text(family = "Palatino Linotype"),
    legend.title     = element_blank(),
    panel.grid.major = element_line(color = rgb(0.7, 0.7, 0.7, 0.4),
      size = 0.1, linetype = "dashed")
  )

gg.rows.spline.rel = dt.rows %>%
  filter(learner == "spline" & Algorithm == "compboost") %>%
  ggplot(aes(x = reorder(n.rows, n.rows), y = rel.factor)) +
  geom_col(position = "dodge", width = 0.1, fill = "#55acee") +
  scale_fill_tech(theme = "twitter") +
  ggtitle("") +
  ylab("") +
  theme(
    panel.background = element_blank(),
    text             = element_text(family = "Palatino Linotype"),
    legend.position  = "none",
    panel.grid.major = element_line(color = rgb(0.7, 0.7, 0.7, 0.4),
      size = 0.1, linetype = "dashed")
  )

# Extracxt the legend from gg.iters.spline:
legend = gtable_filter(ggplotGrob(gg.rows.spline), "guide-box")

# Make custom title:
gtitle = textGrob(label = "Benchmark for Increasing Number of Rows",
  vjust = 0.5, gp = gpar(fontfamily = "Palatino Linotype", fontface = "bold", cex = 1.5))

# Arrange and draw the plot
y.label = textGrob("", rot = 90, vjust = 0.5,
  gp = gpar(fontfamily = "Palatino Linotype"))

x.label = textGrob("Number of Rows", vjust = -0.5,
  gp = gpar(fontfamily = "Palatino Linotype"))

grid.arrange(y.label,
  arrangeGrob(
    gg.rows.linear + theme(legend.position="none") + xlab("") + ylim(0, 250),
    gg.rows.spline + theme(legend.position="none") + ylab("") + xlab("") + ylim(0, 250),
    gg.rows.linear.rel + xlab("") + ylim(0, 10),
    gg.rows.spline.rel + xlab("") + ylim(0, 10),
    layout_matrix = layout.mat,
    top = gtitle,
    bottom = x.label
  ), legend,
  widths = unit.c(unit(2, "lines"), unit(1, "npc") - unit(2, "lines") - legend$width,
    legend$width), nrow=1)
