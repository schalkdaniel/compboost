# ============================================================================ #
#                                                                              #
#                        Algorithms for the Benchmark                          #
#                                                                              #
# ============================================================================ #

# compboost:
# -------------------------

benchmarkCompboost = function (job, data, instance, iters, learner) {

  memorySnap = function () {
    snap = system(command = "top -b -n1 | grep KiB.Mem.*", intern = TRUE)
    snap = strsplit(strsplit(x = snap, split = "KiB Mem : ")[[1]][2], ", ")[[1]]
  
    snap.numbers = as.numeric(gsub("([0-9]+).*$", "\\1", snap))
    names(snap.numbers) = c("total", "free", "used", "buff/cache")
  
    return (snap.numbers)
  }

  gc()

  if (! learner %in% c("spline", "linear")) {
    stop("No valid learner!")
  }

  if (learner == "spline") {

    time = proc.time()
    memory.before = memorySnap()
    cboost = boostSplines(data = instance$data, target = "y", loss = LossQuadratic$new(), iterations = iters, penalty = 2)
    memory.after = memorySnap()
    time = proc.time() - time

  }
  if (learner == "linear") {

    time = proc.time()
    memory.before = memorySnap()
    cboost = boostLinear(data = instance$data, target = "y", loss = LossQuadratic$new(), iterations = iters)
    memory.after = memorySnap()
    time = proc.time() - time

  }

  return (list(time = time, data.dim = dim(instance$data), learner = learner,
    iters = iters, algo = "compboost", memory.before = memory.before, 
    memory.after = memory.after))
}

# mboost:
# -------------------------

benchmarkMboost = function (job, data, instance, iters, learner) {

  memorySnap = function () {
    snap = system(command = "top -b -n1 | grep KiB.Mem.*", intern = TRUE)
    snap = strsplit(strsplit(x = snap, split = "KiB Mem : ")[[1]][2], ", ")[[1]]
  
    snap.numbers = as.numeric(gsub("([0-9]+).*$", "\\1", snap))
    names(snap.numbers) = c("total", "free", "used", "buff/cache")
  
    return (snap.numbers)
  }

  gc()

  if (! learner %in% c("spline", "linear")) {
    stop("No valid learner!")
  }

  getMboostFormula = function (data, target, learner) {
    data.names = setdiff(names(data), target)

    if (learner == "spline") {
      # Ensure the same parameter as compboost:
      myformula = paste0(
        target, " ~ ",
        paste(
          paste0("bbs(", data.names, ", knots = 20, degree = 3, differences = 2, lambda = 2)"),
          collapse = " + "
          )
        )  
    }
    if (learner == "linear") {
      # Ensure the same parameter as compboost:
      myformula = paste0(
        target, " ~ ",
        paste(
          paste0("bols(", data.names, ")"),
          collapse = " + "
          )
        )  
    }

    return (as.formula(myformula))
  }

  if (learner == "spline") {

    time = proc.time()
    memory.before = memorySnap()
    mod.mboost = mboost(getMboostFormula(instance$data, "y", "spline"), data = instance$data, control = boost_control(mstop = iters, nu = 0.05))
    memory.after = memorySnap()
    time = proc.time() - time

  }
  if (learner == "linear") {

    time = proc.time()
    memory.before = memorySnap()
    mod.mboost = mboost(getMboostFormula(instance$data, "y", "linear"), data = instance$data, control = boost_control(mstop = iters, nu = 0.05))
    memory.after = memorySnap()
    time = proc.time() - time

  }

  return (list(time = time, data.dim = dim(instance$data), learner = learner,
    iters = iters, algo = "mboost", memory.before = memory.before, 
    memory.after = memory.after))
}

# mboost fast:
# -------------------------

benchmarkMboostFast = function (job, data, instance, iters, learner) {

  memorySnap = function () {
    snap = system(command = "top -b -n1 | grep KiB.Mem.*", intern = TRUE)
    snap = strsplit(strsplit(x = snap, split = "KiB Mem : ")[[1]][2], ", ")[[1]]
  
    snap.numbers = as.numeric(gsub("([0-9]+).*$", "\\1", snap))
    names(snap.numbers) = c("total", "free", "used", "buff/cache")
  
    return (snap.numbers)
  }

  gc()

  getMboostFormula = function (data, target, learner) {
    data.names = setdiff(names(data), target)

    # Ensure the same parameter as compboost:
    myformula = paste0(
      target, " ~ ",
      paste(
        paste0("bbs(", data.names, ", knots = 20, degree = 3, differences = 2, lambda = 2)"),
        collapse = " + "
        )
      )  
    return (as.formula(myformula))
  }

  if (! learner %in% c("spline", "linear")) {
    stop("No valid learner!")
  }
  if (learner == "spline") {

    time = proc.time()
    memory.before = memorySnap()
    mod.mboost = gamboost(getMboostFormula(instance$data, "y", "spline"), data = instance$data, control = boost_control(mstop = iters, nu = 0.05))
    memory.after = memorySnap()
    time = proc.time() - time

  }
  if (learner == "linear") {

    time = proc.time()
    memory.before = memorySnap()
    mod.mboost = glmboost(target ~ ., data = instance$data, control = boost_control(mstop = iters, nu = 0.05))
    memory.after = memorySnap()
    time = proc.time() - time

  }

  return (list(time = time, data.dim = dim(instance$data), learner = learner,
    iters = iters, algo = ifelse(learner == "spline", "gamboost", "glmboost"), memory.before = memory.before, 
    memory.after = memory.after))
}
