## ---- include=FALSE-----------------------------------------------------------
knitr::opts_chunk$set(collapse = TRUE)
# devtools::load_all()
library(compboost)

options(width = 80)

required.pcks = c("ggplot2")

dependencies = all(
  unlist(lapply(required.pcks, requireNamespace, quietly = TRUE))
)

## -----------------------------------------------------------------------------
# Store train and test data:
df.train = na.omit(titanic::titanic_train)
df.test  = na.omit(titanic::titanic_test)

str(df.train)

## -----------------------------------------------------------------------------
df.train$Survived = factor(df.train$Survived, labels = c("no", "yes"))

# Train and evaluation split for training:
set.seed(1111)

idx.train = sample(x = seq_len(nrow(df.train)), size = 0.6 * nrow(df.train))
idx.eval  = setdiff(seq_len(nrow(df.train)), idx.train)

## -----------------------------------------------------------------------------
cboost = Compboost$new(data = df.train[idx.train, ], target = "Survived",
  loss = LossBinomial$new())

## -----------------------------------------------------------------------------
# Spline base-learner of age:
cboost$addBaselearner("Age", "spline", BaselearnerPSpline)

# Linear base-learner of age (degree = 1 with intercept is default):
cboost$addBaselearner("Age", "linear", BaselearnerPolynomial)

## -----------------------------------------------------------------------------
# Spline base-learner of fare:
cboost$addBaselearner("Fare", "spline", BaselearnerPSpline, degree = 2,
  n.knots = 14, penalty = 10, differences = 2)

## -----------------------------------------------------------------------------
cboost$addBaselearner("Sex", "categorical", BaselearnerPolynomial,
  intercept = FALSE)

## -----------------------------------------------------------------------------
cboost$getBaselearnerNames()

## -----------------------------------------------------------------------------
cboost$addLogger(logger = LoggerTime, use.as.stopper = FALSE, logger.id = "time",
  max.time = 0, time.unit = "microseconds")

