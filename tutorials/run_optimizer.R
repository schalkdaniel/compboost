# Define a 'BaselearnerList':
# ===================================

X = matrix(1:10, ncol = 1)
y = 3 * as.numeric(X) + rnorm(10, 0, 2)

# Create some stupid baselearner (obviously the linear one is the best):
bl.linear = BaselearnerWrapper$new("l1", X, 1)
bl.quadratic = BaselearnerWrapper$new("q1", X, 2)
bl.cubic = BaselearnerWrapper$new("c1", X, 3)
bl.linear2 = BaselearnerWrapper$new("l0", 2 * X, 1)

# Register the learner:
bl.quadratic$RegisterFactory("x")
bl.cubic$RegisterFactory("x")
bl.linear2$RegisterFactory("2*x")
bl.linear$RegisterFactory("x")

# Train the lienar one to compare with the result of the optimizer:
bl.linear$train(y)

# Print all registered baselearner:
printRegisteredFactorys()

# Get best Baselearner:
# ===================================

# Use the greedy algorithm:
getBestBaselearner(y)

bl.linear$GetParameter()

# What happens here? Why aren't we get the linear parameter?
# Register that we have also used 2*x as linear baselearner which basically
# is exactly the same as x. The parameter here is just divided by 2:
bl.linear2$train(y)
bl.linear2$GetParameter()

# Clear the registry:
clearRegisteredFactorys()

# Take a look if all registrys were deleted:
printRegisteredFactorys()
