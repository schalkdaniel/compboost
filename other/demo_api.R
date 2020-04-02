devtools::load_all()
library(mboost)

mydata = na.omit(hflights::hflights)
mydata = rbind(mydata, mydata, mydata)

mydata$UniqueCarrier = NULL
mydata$TailNum = NULL
mydata$Origin = NULL
mydata$Dest = NULL

time1 = proc.time()
cboost1 = boostSplines(data = mydata, target = "DepDelay", loss = LossQuadratic$new(), iterations = 1000L, learning_rate = 0.1, use_binning = FALSE, trace = 250L)
time1 = proc.time() - time1

time2 = proc.time()
cboost2 = boostSplines(data = mydata, target = "DepDelay", loss = LossQuadratic$new(), iterations = 1000L, learning_rate = 0.1, use_binning = TRUE, trace = 250L)
time2 = proc.time() - time2

form = as.formula(paste0("DepDelay ~ ", paste(paste0("bbs(", names(mydata)[names(mydata) != "DepDelay"], ", degree = 3, knots = 20)"), sep = "", collapse = " + ")))

time3 = proc.time()
mod_mboost = mboost(form, data = mydata, control = boost_control(mstop = 1000L))
time3 = proc.time() - time3

gg1 = cboost1$plotBlearnerTraces()
gg1_feat = cboost1$plot("Distance_spline", iters = c(125, 250, 500, 750, 1000))

gg2 = cboost2$plotBlearnerTraces()
gg2_feat = cboost2$plot("Distance_spline", iters = c(125, 250, 500, 750, 1000))

gridExtra::grid.arrange(gg1,gg2, ncol = 2L)
gridExtra::grid.arrange(gg1_feat, gg2_feat, ncol = 2L)


# Memory in KB
used_memory = data.frame(
  before_start = c(3123708, 3927388),
  after_250 = c(3991912, 4165208),
  after_750 = c(3945364, 4155552),
  after_finish = c(3920916, 4162068)
)

plt_data = data.frame(
  method = c("Complete Features", "Binned Features"),
  runtime = c(time1[3], time2[3]),
  memory = apply(used_memory, 1, function (x) { median(x[-1] - x[1]) / 1000 })
)


library(ggplot2)

gg =
ggplot(plt_data, aes(x = method, y = memory, fill = method)) +
  geom_bar(stat = "identity") +
  geom_text(aes(y = 0, label = paste0(round(runtime, 2), " Sek.")), color = "white", vjust = -1, size = 7) +
  scale_fill_brewer(palette = "Set1") +
  labs(fill = "") +
  ggtitle("Used memory during fitting process") +
  ylab("Memory in MB") +
  xlab("") +
  theme_minimal()

ggsave(gg, filename = "gg_binning.png", height = 4, width = 7)





src = "
arma::SpMat<unsigned int> dummyMat(const arma::Row<unsigned int>& classes)
{
  arma::Mat<unsigned int> locations(2, classes.size(), arma::fill::zeros);
  locations.row(0) = arma::linspace<arma::Row<unsigned int>>(0, classes.size() - 1, classes.size());
  locations.row(1) = classes;

  arma::Col<unsigned int> values(classes.size(), arma::fill::ones);
  arma::SpMat<unsigned int> binary_mat(locations, values);

  return binary_mat;
}
"

Rcpp::cppFunction(code = src, depends = "RcppArmadillo")

cls = sample(seq_len(5), 10, replace = TRUE)
dummyMat(cls - 1)
