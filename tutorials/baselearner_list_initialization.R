# Define a 'BaselearnerList':
# ===================================

X = matrix(1:10, ncol = 1)
y = 3 * as.numeric(X) + rnorm(10, 0, 2)

# Create new object (Note that we call a polynomial with degree 1):
bl.linear = BaselearnerWrapper$new("l1", X, "var1", 1)

# Create new object (Note that we call a polynomial with degree 2):
bl.quadratic = BaselearnerWrapper$new("q1", X, "var1", 2)

bl.linear$RegisterFactory("x")
bl.quadratic$RegisterFactory("x1")

printRegisteredFactorys()
clearRegisteredFactorys()
printRegisteredFactorys()

bl.linear$RegisterFactory("testItest")

printRegisteredFactorys()
