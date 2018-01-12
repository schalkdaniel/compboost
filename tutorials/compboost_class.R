# ============================================================================ #
#                                  compboost                                   #
# ============================================================================ #

# We want to model the miles per gallon (mpg) dependent of the horse power 
# (hp) as linear and quadratic learner and the weight (wt) just as linear 
# learner.

# We want to use a 500 iterations as maxium, a lerning rate of 0.05 and 
# quadratic loss.

# Prepare Data:
# -------------

df = mtcars

# Create new variable to check the polynomial baselearner with degree 2:
df$hp2 = df[["hp"]]^2

# Data for compboost:
X.hp = as.matrix(df[["hp"]], ncol = 1)
X.wt = as.matrix(df[["wt"]], ncol = 1)

y = df[["mpg"]]

# Hyperparameter for the algorithm:
learning.rate = 0.05
iter.max = 500


# Prepare compboost:
# ------------------

# Create new linear baselearner of hp and wt:
linear.factory.hp = PolynomialFactory$new(X.hp, "hp", 1)
linear.factory.wt = PolynomialFactory$new(X.wt, "wt", 1)

# Create new quadratic baselearner of hp:
quadratic.factory.hp = PolynomialFactory$new(X.hp, "hp", 2)

# Create new factory list:
factory.list = FactoryList$new()

# Register factorys:
factory.list$registerFactory(linear.factory.hp)
factory.list$registerFactory(linear.factory.wt)
factory.list$registerFactory(quadratic.factory.hp)

# Print the registered factorys:
factory.list$printRegisteredFactorys()

# Use quadratic loss:
loss.quadratic = QuadraticLoss$new()

# Run compboost:
# --------------

# Initialize object (Response, learning rate, maximal iterations, stop if all
# stopper are fulfilled?, maximal microseconds, factory list):
cboost = Compboost$new(y, learning.rate, iter.max, TRUE, 0, factory.list, loss.quadratic)

# Train the model:
cboost$train()

# Get vector selected baselearner:
cboost$getSelectedBaselearner()
# cboost$GetModelFrame()
cboost$getLoggerData()
