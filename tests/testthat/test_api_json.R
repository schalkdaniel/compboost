context("Test if JSON loading via the R6 API is working")

test_that("Basic save and load works with all loss and optimizer combinations", {
  file = "cboost.json"

  losses = c(LossQuadratic$new(), LossQuantile$new(0.2), LossAbsolute$new(), LossHuber$new())

  optimizers = c(OptimizerCoordinateDescent$new(1),
    OptimizerCoordinateDescentLineSearch$new(1),
    OptimizerCosineAnnealing$new(1),
    OptimizerAGBM$new(0.1, 1))

  lo_grid = expand.grid(l = losses, o = optimizers)

  for (i in seq_len(nrow(lo_grid))) {
    l = lo_grid$l[[i]]
    o = lo_grid$o[[i]]

    iters = 100L
    if (o$getOptimizerType() == "agbm") {
      iters = 100L
    }

    set.seed(3141)
    cb = expect_output(boostSplines(iris, "Sepal.Length", iterations = iters, loss = l, optimizer = o))
    cboost = testCboostJsonAPI(cb)

    #message(sprintf("Done with %s/%s: l = %s, o = %s", i, nrow(lo_grid), l$getLossType(), o$getOptimizerType()))

    invisible(lapply(unique(cboost$getSelectedBaselearner()), function(bln) {
      expect_equal(class(expect_silent(plotBaselearner(cboost, bln))), c("gg", "ggplot"))
    }))
    selbl = unique(cboost$getSelectedBaselearner())
    selfn = vapply(cboost$baselearner_list[selbl], function(f) unique(f$factory$getFeatureName()), character(1), USE.NAMES = FALSE)
    invisible(lapply(selfn, function(f) {
      expect_equal(class(expect_silent(plotPEUni(cboost, f))), c("gg", "ggplot"))
    }))
  }

  file.remove(file)
})

test_that("Basic save and load works for centered base learner with all loss and optimizer combinations", {
  file = "cboost.json"

  losses = c(LossQuadratic$new(), LossQuantile$new(0.2), LossAbsolute$new(), LossHuber$new())

  optimizers = c(OptimizerCoordinateDescent$new(1),
    OptimizerCoordinateDescentLineSearch$new(1),
    OptimizerCosineAnnealing$new(1),
    OptimizerAGBM$new(0.1, 1))

  lo_grid = expand.grid(l = losses, o = optimizers)

  for (i in seq_len(nrow(lo_grid))) {
    l = lo_grid$l[[i]]
    o = lo_grid$o[[i]]

    cb = expect_silent(Compboost$new(data = iris, target = "Sepal.Length", loss = l, optimizer = o))
    invisible(lapply(c("Sepal.Width", "Petal.Length", "Petal.Width"), function(fn) {
      expect_silent(cb$addComponents(fn))
    }))
    expect_silent(cb$addBaselearner("Species", "ridge", BaselearnerCategoricalRidge))
    expect_output(cb$train(100))

    cboost = testCboostJsonAPI(cb)
    invisible(lapply(unique(cboost$getSelectedBaselearner()), function(bln) {
      expect_equal(class(expect_silent(plotBaselearner(cboost, bln))), c("gg", "ggplot"))
    }))
    selbl = unique(cboost$getSelectedBaselearner())
    selfn = vapply(cboost$baselearner_list[selbl], function(f) unique(f$factory$getFeatureName()), character(1), USE.NAMES = FALSE)
    invisible(lapply(selfn, function(f) {
      expect_equal(class(expect_silent(plotPEUni(cboost, f))), c("gg", "ggplot"))
    }))
  }

  file.remove(file)
})

test_that("Basic save and load works for tensor base learner with all loss and optimizer combinations", {
  file = "cboost.json"

  losses = c(LossQuadratic$new(), LossQuantile$new(0.2), LossAbsolute$new(), LossHuber$new())

  optimizers = c(OptimizerCoordinateDescent$new(1),
    OptimizerCoordinateDescentLineSearch$new(1),
    OptimizerCosineAnnealing$new(1),
    OptimizerAGBM$new(0.1, 1))

  lo_grid = expand.grid(l = losses, o = optimizers)

  for (i in seq_len(nrow(lo_grid))) {
    l = lo_grid$l[[i]]
    o = lo_grid$o[[i]]
 0
    cb = expect_silent(Compboost$new(data = iris, target = "Sepal.Length", loss = l, optimizer = o))
    expect_silent(cb$addTensor("Species", "Sepal.Width", n_knots = 5))
    expect_silent(cb$addTensor("Petal.Length", "Petal.Width", n_knots = 5))
    expect_output(cb$train(100))

    cboost = testCboostJsonAPI(cb)
    invisible(lapply(unique(cboost$getSelectedBaselearner()), function(bln) expect_error(plotBaselearner(cboost, bln))))

    invisible(lapply(unique(cboost$getSelectedBaselearner()), function(bln) {
      expect_equal(class(expect_silent(plotTensor(cboost, bln, npoints = 10L))), c("gg", "ggplot"))
    }))
  }

  file.remove(file)
})
