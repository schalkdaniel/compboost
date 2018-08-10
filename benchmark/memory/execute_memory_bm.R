source("memory/memory_helper.R")

library(compboost)
library(mboost)

set.seed(314159)
iters = 1500

sim.data = simulateData(n = 50000, vars = 1000)

mboost.formula.linear = getMboostFormula(sim.data, "y", "linear")
mboost.formula.spline = getMboostFormula(sim.data, "y", "spline")

# Run compboost:
# ------------------------------------

# Linear base-learner (~9 Min):
cboost = boostLinear(data = sim.data, target = "y", loss = LossQuadratic$new(), iterations = iters)
# trackMemory(how.long = 9 * 60, trace = TRUE, save = "memory/benchmark_files/cboost_linear.rds")

# Spline base-learner (~40 Min):
cboost = boostSplines(data = sim.data, target = "y", loss = LossQuadratic$new(), iterations = iters)
# trackMemory(how.long = 40 * 60, trace = TRUE, save = "memory/benchmark_files/cboost_spline.rds")

# Run mboost:
# ------------------------------------

# Linear base-learner (~50 Min):
mod.mboost = mboost(mboost.formula.linear, data = sim.data, control = boost_control(mstop = iters, nu = 0.05))
# trackMemory(how.long = 50 * 60, trace = TRUE, save = "memory/benchmark_files/mboost_linear.rds")

# Spline base-learner (~80 Min):
mod.mboost = mboost(mboost.formula.spline, data = sim.data, control = boost_control(mstop = iters, nu = 0.05))
# trackMemory(how.long = 80 * 60, trace = TRUE, save = "memory/benchmark_files/mboost_spline.rds")


# Run glmboost/gamboost:
# ------------------------------------

# Linear base-learner (~4 Min):
mod.mboost = glmboost(y ~ ., data = sim.data, control = boost_control(mstop = iters, nu = 0.05))
# trackMemory(how.long = 4 * 60, trace = TRUE, save = "memory/benchmark_files/glmboost.rds")

# Spline base-learner (~ 80 Min):
mod.mboost = gamboost(mboost.formula.spline, data = sim.data, control = boost_control(mstop = iters, nu = 0.05))
# trackMemory(how.long = 80 * 60, trace = TRUE, save = "memory/benchmark_files/gamboost.rds")
