nsim = 1000

outliers = TRUE

x = runif(nsim, 0, 10)
y = 3 + 2 * sin(x) + rnorm(nsim, 0, 1)

if (outliers) {
  noutlier = 20
  outlier_mean = 1e6
  outlier_idx = sample(nsim, noutlier)
  y[outlier_idx] = sample(x = c(-1, 1), size = noutlier, replace = TRUE) * rnorm(noutlier, outlier_mean, 1)
}

plot(x = x, y = y)

df = data.frame(x = x, y = y)

cboost = boostSplines(data = df, target = "y", loss = LossQuadratic$new(), iterations = 1000)
cboost_10 = boostSplines(data = df, target = "y", loss = LossQuantile$new(0.1), iterations = 1000)
cboost_50 = boostSplines(data = df, target = "y", loss = LossQuantile$new(0.5), iterations = 1000)
cboost_90 = boostSplines(data = df, target = "y", loss = LossQuantile$new(0.9), iterations = 1000)

df_pred = data.frame(
  feat = x,
  target = y,
  mean = cboost$predict(),
  quantile_10 = cboost_10$predict(),
  quantile_50 = cboost_50$predict(),
  quantile_90 = cboost_90$predict()
)
df_pred = tidyr::gather(data = df_pred, key = "Estimator", value = "pred", mean:quantile_90)

library(ggplot2)

gg1 = ggplot() +
  geom_point(data = df_pred, aes(x = feat, y = target)) +
  geom_line(data = df_pred, aes(x = feat, y = pred, color = Estimator, linetype = Estimator), size = 2, show.legend = FALSE) +
  xlab("Feature") +
  ylab("Target") +
  ggthemes::theme_tufte() +
  scale_color_brewer(palette = "Set1") +
  ggtitle("Full Target Range WITH Outliers")

gg2 = ggplot() +
  geom_point(data = df_pred, aes(x = feat, y = target)) +
  geom_line(data = df_pred, aes(x = feat, y = pred, color = Estimator, linetype = Estimator), size = 2) +
  ylim(-2, 8) +
  xlab("Feature") +
  ylab("") +
  ggthemes::theme_tufte() +
  scale_color_brewer(palette = "Set1") +
  ggtitle("Real Target Range WITHOUT Outliers")

final_plot = gridExtra::grid.arrange(gg1, gg2, ncol = 2)
ggsave(plot = final_plot, filename = "cboost_quantile_regression.pdf")



par(mfrow = c(1,2))

plot(x = x, y = y, main = "bla")
lines(x = x[order(x)], y = preds[order(x)], col = "blue", lty = 1, lwd = 2)
lines(x = x[order(x)], y = preds_50[order(x)], col = "red", lty = 1, lwd = 2)
lines(x = x[order(x)], y = preds_10[order(x)], col = "red", lty = 2, lwd = 2)
lines(x = x[order(x)], y = preds_90[order(x)], col = "red", lty = 2, lwd = 2)


plot(x = x, y = y, ylim = c(-2, 8))
lines(x = x[order(x)], y = preds[order(x)], col = rgb(0, 0, 1, 0.2), lty = 1, lwd = 2)
lines(x = x[order(x)], y = preds_50[order(x)], col = "red", lty = 1, lwd = 2)
lines(x = x[order(x)], y = preds_10[order(x)], col = "red", lty = 2, lwd = 2)
lines(x = x[order(x)], y = preds_90[order(x)], col = "red", lty = 2, lwd = 2)

par(mfrow = c(1,1))