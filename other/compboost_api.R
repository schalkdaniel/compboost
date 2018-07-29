Rcpp::compileAttributes()
roxygen2::roxygenize()
# devtools::document()
devtools::load_all()




# Create categorical feature:
mtcars$mpg_cat = ifelse(mtcars$mpg > 15, "A", "B")

cboost = Compboost$new(mtcars, "mpg", loss = LossQuadratic$new())

# Should throw an error:
cboost$train(10)

cboost$addBaselearner("wt", "spline", BaselearnerPSplineFactory, degree = 3, 
	knots = 10, penalty = 2, differences = 2)

cboost$addBaselearner("mpg_cat", "linear", BaselearnerPolynomialFactory, degree = 1, intercept = FALSE)

# Error should apprear:
cboost$addBaselearner(c("hp", "wt"), "spline", BaselearnerPSplineFactory, degree = 3, 
	knots = 10, penalty = 2, differences = 2)

cboost$addBaselearner(c("hp", "wt"), "linear", BaselearnerPolynomialFactory, degree = 1, intercept = TRUE)
cboost$addBaselearner("hp", "quadratic", BaselearnerPolynomialFactory, degree = 2, intercept = TRUE)

cboost$addLogger(logger = LoggerTime, use.as.stopper = FALSE, logger.id = "time", max.time = 0, time.unit = "microseconds")

cboost$bl.factory.list
cboost$getBaselearnerNames()
cboost$bl.factory.list$getRegisteredFactoryNames()

cboost$train(4000)
str(cboost$risk())

gc()

cboost$train(2000)
str(cboost$risk())

cboost$train(12000)
str(cboost$risk())

cboost$predict(mtcars[1:2,])

a = cboost$selected()
a[1:20]
table(a)
cboost$getBaselearnerNames()

cboost$coef()

cboost$plot("wt_spline")
cboost$plot("wt_spline", iters = c(100, 500, 1000, 2000))
cboost$plot("hp_quadratic")
cboost$plot("hp_quadratic", iters = c(100, 500, 1000, 2000))
cboost$plot("hp_quadratic", iters = c(1000, 2000))


cboost$plot("wt_spline", iters = c(100, 500, 1000, 2000, 10000), from = 2, to = 4) +
labs(title = "Effect of Weight", subtitle = "Additive contribution of linear predictor") +
theme_tufte() + 
scale_color_brewer(palette = "Spectral")

cboost$plot("hp_quadratic", from = 0)
cboost$plot("hp_quadratic", to = 10)
cboost$plot("hp_quadratic", from = 0, to = 10)



# Check for other errors:
cboost$plot("bla")
cboost$plot("hp_wt_linear")
cboost$plot("mpg_cat_B_linear")


identical(cboost$predict(), cboost$predict(mtcars))
cboost$model$getEstimatedParameter()





cboost$train(500)

cboost$coef()
gg = cboost$plot("wt_spline")

all.equal(cboost$predict(), cboost$predict(mtcars))







# Taking predictions:
pred.list = list()
pred.list[[1]] = InMemoryData$new(
	as.matrix(seq(from = min(mtcars$hp), to = max(mtcars$hp), length.out = 100)), 
	"hp"
	)
pred.list[[2]] = InMemoryData$new(
	as.matrix(seq(from = min(mtcars$wt), to = max(mtcars$wt), length.out = 100)), 
	"wt"
	)

pred = cboost$model$predict(pred.list)

cboost$model$getOffset()

cboost$predict()


## Test intercept:
## --------------------------------
data.mat = cbind(1:10)

# Create new data object:
data.source = InMemoryData$new(data.mat, "my.data.name")

data.target1 = InMemoryData$new()
data.target2 = InMemoryData$new()


lin.factory = BaselearnerPolynomialFactory$new(data.source, data.target1, degree = 2, intercept = TRUE)
lin.factory.intercept = BaselearnerPolynomialFactory$new(data.source, data.target2, "quadratic", degree = 2, intercept = TRUE)
# Get the transformed data:
lin.factory
lin.factory.intercept

lin.factory$getData()
lin.factory.intercept$getData()


devtools::load_all()

# Define custom "loss function"
aucLoss = function (truth, response) {
  # Convert response on f basis to probs using sigmoid:
	probs = 1 / (1 + exp(-response))

  #  Calculate AUC:
	mlr:::measureAUC(probabilities = probs, truth = truth, negative = -1, positive = 1) 
}

# Define also gradient and constant initalization since they are necessary for
# the custom loss:
gradDummy = function (trutz, response) { return (NA) }
constInitDummy = function (truth, response) { return (NA) }

# Define loss:
auc.loss = LossCustom$new(aucLoss, gradDummy, constInitDummy)


mtcars$mpg_bin = ifelse(mtcars$mpg > 15, -1, 1)
idx.train = sample(seq_len(nrow(mtcars)), nrow(mtcars) * 0.6)
idx.test = setdiff(seq_len(nrow(mtcars)), idx.train)

cboost = Compboost$new(mtcars[idx.train,], "mpg_bin", loss = LossBinomial$new())

cboost$addBaselearner("wt", "spline", BaselearnerPSplineFactory, degree = 3, 
	knots = 10, penalty = 2, differences = 2)

cboost$addLogger(logger = LoggerTime, use.as.stopper = FALSE, logger.id = "time", max.time = 0, time.unit = "microseconds")
cboost$addLogger(logger = LoggerOobRisk, use.as.stopper = FALSE, logger.id = "auc_oob",
	auc.loss, 0.01, cboost$prepareData(mtcars[idx.test, ]), mtcars[idx.test, "mpg_bin"])
cboost$addLogger(logger = LoggerInbagRisk, use.as.stopper = FALSE, logger.id = "auc_inbag",
	auc.loss, 0.01)
cboost$addLogger(logger = LoggerOobRisk, use.as.stopper = FALSE, logger.id = "risk_oob",
	LossBinomial$new(), 0.01, cboost$prepareData(mtcars[idx.test, ]), mtcars[idx.test, "mpg_bin"])

cboost$train(2000)

str(cboost$model$getLoggerData())

plot.df = data.frame(
	AUC = c(cboost$model$getLoggerData()[[2]][,2],
		cboost$model$getLoggerData()[[2]][,3]),
	Iteration = rep(1:2000, 2),
	Risk = rep(c("Inbag", "OOB"), each = 2000)
)

library(ggplot2)
ggplot(plot.df, aes(Iteration, AUC, color = Risk)) + geom_line()