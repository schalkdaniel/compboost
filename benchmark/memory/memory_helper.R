simulateData = function (n, vars) {

	requireNamespace("mvtnorm")

  #create beta distributed correlations
  corrs = rbeta(n = (vars * (vars - 1))/2, shape1 = 1, shape2 = 8)
  corrs = sample(c(-1, 1), size = length(corrs), replace = TRUE) * corrs

  sigma = matrix(1, nrow = vars, ncol = vars)
  sigma[upper.tri(sigma)] = corrs
  sigma[lower.tri(sigma)] = t(sigma)[lower.tri(sigma)]

  data = as.data.frame(mvtnorm::rmvnorm(n = n, sigma = sigma, method = "svd"))

  betas = runif(vars + 1, min = -2, max = 2)
  data$y = rnorm(n = n, mean = as.matrix(cbind(1, data[,1:vars])) %*% betas)

  #return (list(data = data, betas = betas))
  return (data)
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

memorySnap = function () {
	if (Sys.info()["sysname"] != "Linux") {
		stop("Sorry, but tracking performance (with this function) is just possible on a Linux machine.")
	}
  snap = system(command = "top -b -n1 | grep KiB.Mem.*", intern = TRUE)
  snap = strsplit(strsplit(x = snap, split = "KiB Mem : ")[[1]][2], ", ")[[1]]

  snap.numbers = as.numeric(gsub("([0-9]+).*$", "\\1", snap))
  names(snap.numbers) = c("total", "free", "used", "buff/cache")

  return (snap.numbers)
}

trackMemory = function (steps = 2, how.long = 3600, trace = FALSE, save = NULL)
{
	if (Sys.info()["sysname"] != "Linux") {
		stop("Sorry, but tracking performance (with this function) is just possible on a Linux machine.")
	}

  memory.out = data.frame(timestamp = character(0), second = numeric(0), 
  	free.memory = integer(0), used.memory = integer(0), stringsAsFactors = FALSE)

  time.initial = Sys.time()
  time.diff = 0

	while (time.diff < how.long) {

		memory.snapshot = memorySnap()
		time.snapshot   = Sys.time()

		memory.out = rbind(memory.out, data.frame(
			timestamp = as.character(time.snapshot), second = as.numeric(time.diff), 
  	  free.memory = memory.snapshot[["free"]], used.memory = memory.snapshot[["used"]], 
  	  stringsAsFactors = FALSE
		))
		# memory.out[i, "timestamp"]   = as.character(time.snapshot)
		# memory.out[i, "second"]      = as.numeric(time.snapshot - time.initial)
		# memory.out[i, "free.memory"] = memory.snapshot[["free"]]
		# memory.out[i, "used.memory"] = memory.snapshot[["used"]]

		if (trace) {
			df.string = capture.output(memory.out[nrow(memory.out), ])
			if (time.diff > 0) {
		  	cat(paste(df.string[2], collapse = "\n"), "\n")
		  } else {
		  	cat(paste(df.string, collapse = "\n"), "\n")
		  }
		}
	  Sys.sleep(steps)

	  time.diff        = time.snapshot - time.initial
		units(time.diff) = "secs"
	}
	if (! is.null(save)) {
		save(list = "memory.out", file = save)
	}
	return (invisible(memory.out))
}

# snap = memorySnap()
# # trackMemory(steps = 1, how.long = 10)
# trackMemory(steps = 1, how.long = 10, trace = TRUE)