# Install the package:
# ====================

library(Rcpp)

compileAttributes()
devtools::load_all()

# Define a loss object with default loss:
# =======================================

# Create new object:
lo = LossWrapper$new()

# Calculate the loss for two vectors. This vectors are acting
# later as real values and the prediction:
lo$CalcLoss(1:10, 2:11)

# Calculate the gradient for this vectors:
lo$CalcGradient(1:10, 2:11)

# Calculate the constant initialization for the loss:
lo$ConstantInitializer(1:10)

# Get the name of the used loss:
lo$GetLossName()

# Define a loss object with predefined loss:
# ==========================================

# Create new object:
lo = LossWrapper$new("absolute")

# Same as above, calculate the loss:
lo$CalcLoss(1:10, rep(5, 10))

# Calculate the gradient:
lo$CalcGradient(1:10, rep(5, 10))

# Calculate the constant initialization for the loss:
lo$ConstantInitializer(1:10)

# Get the name of the used loss:
lo$GetLossName()

# Define a custom loss with R functions:
# ======================================

# Define an own loss function:
lossOwn = function (x, y)
{
  return((x - y)^4)
}

#  Define the corresponding gradient:
gradOwn = function (x, y)
{
  return((x - y)^3 / 4)
}

# Define the corresponding initializer:
initOwn = function (x)
{
  optimum = optim(
    par = c(constant = 0),
    fn  = function (x, constant) { return(sum(lossOwn(x = x, y = constant))) },
    gr  = function (x, constant) { return(sum(gradOwn(x = x, y = constant))) },
    x   = x,
    method = "BFGS"
  )$par
  return(rep(optimum, length(x)))
}

# Two variables to test the functionality:
set.seed(pi)
x = rnorm(10)
y = rnorm(10)

# Now define the custom loss. It is sufficient to call the constructor
# with the R functions within:
lo = LossWrapper$new("power4loss", lossOwn, gradOwn, initOwn)

# Calculate the loss and check with the defined 'lossOwn' function:
lo$CalcLoss(x, y)
lossOwn(x, y)

# Calculate the gradient and check with the defined 'gradOwn' function:
lo$CalcGradient(x, y)
gradOwn(x, y)

# Calculate the initializer and check with the defined 'initOwn' function:
lo$ConstantInitializer(x)
initOwn(x)

# Get the name of the used loss:
lo$GetLossName()
