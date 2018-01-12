# ============================================================================ #
#                           Baselearner (Factorys)                             #
# ============================================================================ #

# The baselearner are the key of model based boosting. Therefore we have 
# implemented a way of using the baselearner without any compboost object.
# Note that this is just thought to be used for testing! 

# Nevertheless, it is possible to train a baselearner and get predictions 
# for this specific one. This is not directly accessible, instead we have
# to create a factory. This factory then creates one object of the. This must
# be done:
# 
#   1. The first createInstance also instantiates the data
#   2. We have now a baselearner on which we can do some testing
#   
# Note that this does not result in less performance, since we just create 
# the object. The data are created one way or another.

# Some sample data:
# -----------------

X = matrix(1:10, ncol = 1)
y = 3 * as.numeric(X) + rnorm(10, 0, 2)

# Define a linear baselearner factory:
# ------------------------------------

# Create new factory (Note that we call a polynomial with degree 1). Note that
# the data identifier (third argument) has to be exactly the same string as
# the corresponding colname in a later dataframe:
linear.factory = PolynomialFactory$new(X, "my_variable_name", 1)

# Use the test baselearner within the factory:
# --------------------------------------------

# Now we can train the inherent baselearner and do a simple prediction.

# Train baselearner:
linear.factory$testTrain(y)

# Get estimated parameter:
linear.factory$testGetParameter()

# This should be the same as in a linear model without intercept:
lm(y ~ 0 + X)$coef

# Now we can call the prediction:
linear.factory$testPredict()

# This should be the same as in the linear model to:
mod = lm(y ~ 0 + X)$fitted.values

# We can access the data used for training:
linear.factory$getData()
