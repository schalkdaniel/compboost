context("BlearnerFactoryList of 'compboost'")

test_that("factory list works", {

  df = mtcars

  # Create new variable to check the polynomial baselearner with degree 2:
  df$hp2 = df[["hp"]]^2

  # Data for compboost:
  X.hp = as.matrix(df[["hp"]], ncol = 1)

  # Use wt as linear baselearner with intercept:
  X.wt = as.matrix(df[["wt"]], ncol = 1)
  X.wt = cbind(1, X.wt)
  
  data.source.hp = InMemoryData$new(X.hp, "hp")
  data.source.wt = InMemoryData$new(X.wt, "wt")
  
  data.target1 = InMemoryData$new()
  data.target2 = InMemoryData$new()
  data.target3 = InMemoryData$new()
  data.target4 = InMemoryData$new()

  # Prepare compboost:
  # ------------------

  # Create new linear baselearner of hp and wt:
  linear.factory.hp = PolynomialBlearnerFactory$new(data.source.hp, data.target1, 1, FALSE)
  linear.factory.wt = PolynomialBlearnerFactory$new(data.source.wt, data.target2, 1, FALSE)

  # Create new quadratic baselearner of hp:
  quadratic.factory.hp = PolynomialBlearnerFactory$new(data.source.hp, data.target3, 2, FALSE)

  # Create new polynomial baselearner to the power of 5:
  pow5.factory.hp = PolynomialBlearnerFactory$new(data.source.hp, data.target4, 5)

  # Create new factory list:
  factory.list = BlearnerFactoryList$new()

  # Register factorys:
  factory.list$registerFactory(linear.factory.hp)
  factory.list$registerFactory(linear.factory.wt)
  factory.list$registerFactory(quadratic.factory.hp)
  factory.list$registerFactory(pow5.factory.hp)

  factory.names = c(
    "hp: polynomial with degree 1",
    "wt: polynomial with degree 1 x1",
    "wt: polynomial with degree 1 x2",
    "hp: polynomial with degree 2",
    "hp: polynomial with degree 5"
  )
  model.frame = cbind(
    X.hp, X.wt, X.hp^2, X.hp^5
  )

  # The underlying hash map of compboost sorts the entrys by key (name):
  model.frame   = model.frame[, order(factory.names)]
  factory.names = sort(factory.names)

  # Test:
  # -------

  expect_equal(factory.list$getModelFrame()$colnames, factory.names)
  expect_equal(factory.list$getModelFrame()$model.frame, model.frame)
  expect_equal(factory.list$getNumberOfRegisteredFactories(), 4)

  factory.list$clearRegisteredFactories()

  expect_equal(factory.list$getNumberOfRegisteredFactories(), 0)

})
