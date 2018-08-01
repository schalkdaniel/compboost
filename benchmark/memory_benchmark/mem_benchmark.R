# ============================================================================ #
#                                                                              #
#                       Script to run the Memory Benchmark                     #
#                                                                              #
# ============================================================================ #

source("mem_benchmark/functions.R")

library(mboost)
library(compboost)

# Base-Line:
# ----------------------------

mydata = simData(n = 2000, p = 1000, seed = 123)

### linear base-learner with 1000 iterations:

# force mboost to use dense matrix:
options(mboost_useMatrix = FALSE)

memBenchmarkCompboost(mydata = mydata, iters = 1000, learner = "linear")
memBenchmarkMboost(mydata = mydata, iters = 1000, learner = "linear")

# Use mboost with potentially sparse matrices:
options(mboost_useMatrix = TRUE)

memBenchmarkCompboost(mydata = mydata, iters = 1000, learner = "linear")
memBenchmarkMboost(mydata = mydata, iters = 1000, learner = "linear")

### spline base-learner with 1000 iterations:

# force mboost to use dense matrix:
options(mboost_useMatrix = FALSE)

memBenchmarkCompboost(mydata = mydata, iters = 1000, learner = "spline")
memBenchmarkMboost(mydata = mydata, iters = 1000, learner = "spline")

# Use mboost with potentially sparse matrices:
options(mboost_useMatrix = TRUE)

memBenchmarkCompboost(mydata = mydata, iters = 1000, learner = "spline")
memBenchmarkMboost(mydata = mydata, iters = 1000, learner = "spline")

# Using 5000 Iterations:
# ----------------------------

### linear base-learner with 5000 iterations:

# force mboost to use dense matrix:
options(mboost_useMatrix = FALSE)

memBenchmarkCompboost(mydata = mydata, iters = 5000, learner = "linear")
memBenchmarkMboost(mydata = mydata, iters = 5000, learner = "linear")

# Use mboost with potentially sparse matrices:
options(mboost_useMatrix = TRUE)

memBenchmarkCompboost(mydata = mydata, iters = 5000, learner = "linear")
memBenchmarkMboost(mydata = mydata, iters = 5000, learner = "linear")

### spline base-learner with 5000 iterations:

# force mboost to use dense matrix:
options(mboost_useMatrix = FALSE)

memBenchmarkCompboost(mydata = mydata, iters = 5000, learner = "spline")
memBenchmarkMboost(mydata = mydata, iters = 5000, learner = "spline")

# Use mboost with potentially sparse matrices:
options(mboost_useMatrix = TRUE)

memBenchmarkCompboost(mydata = mydata, iters = 5000, learner = "spline")
memBenchmarkMboost(mydata = mydata, iters = 5000, learner = "spline")


# Using 2000 Base-Learner:
# ----------------------------


mydata = simData(n = 2000, p = 2000, seed = 123)

### linear base-learner with 2000 base-learner:

# It isn't possible to run mboost with 2000 base-learner on this machine!

# # force mboost to use dense matrix:
# options(mboost_useMatrix = FALSE)

memBenchmarkCompboost(mydata = mydata, iters = 1000, learner = "linear")
# memBenchmarkMboost(mydata = mydata, iters = 1000, learner = "linear")

# # Use mboost with potentially sparse matrices:
# options(mboost_useMatrix = TRUE)
#
# memBenchmarkMboost(mydata = mydata, iters = 1000, learner = "linear")

### spline base-learner with 2000 base-learner:

# # force mboost to use dense matrix:
# options(mboost_useMatrix = FALSE)

memBenchmarkCompboost(mydata = mydata, iters = 1000, learner = "spline")
# memBenchmarkMboost(mydata = mydata, iters = 1000, learner = "spline")

# # Use mboost with potentially sparse matrices:
# options(mboost_useMatrix = TRUE)
#
# memBenchmarkMboost(mydata = mydata, iters = 1000, learner = "spline")



# 50000 Observations:
# ----------------------------

mydata = simData(n = 50000, p = 1000, seed = 123)

### linear base-learner with 50000 observations:

# force mboost to use dense matrix:
options(mboost_useMatrix = FALSE)

memBenchmarkCompboost(mydata = mydata, iters = 1000, learner = "linear")
memBenchmarkMboost(mydata = mydata, iters = 1000, learner = "linear")

# Use mboost with potentially sparse matrices:
options(mboost_useMatrix = TRUE)

memBenchmarkMboost(mydata = mydata, iters = 1000, learner = "linear")

### spline base-learner with 50000 observations:

# force mboost to use dense matrix:
options(mboost_useMatrix = FALSE)

memBenchmarkCompboost(mydata = mydata, iters = 1000, learner = "spline")
memBenchmarkMboost(mydata = mydata, iters = 1000, learner = "spline")

# Use mboost with potentially sparse matrices:
options(mboost_useMatrix = TRUE)

memBenchmarkMboost(mydata = mydata, iters = 1000, learner = "spline")
