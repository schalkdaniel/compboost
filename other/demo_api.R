devtools::load_all()

cboost = boostSplines(data = mtcars, target = "mpg", loss = LossQuadratic$new(), iterations = 1000L, learning_rate = 0.01)

cboost$plotBlearnerTraces()
cboost$plot("hp_spline", iters = c(125, 250, 500, 750, 1000))
