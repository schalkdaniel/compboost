context("FactoryList of 'compboost'")

test_that("factory list works", {
  
  df = mtcars
  
  # Create new variable to check the polynomial baselearner with degree 2:
  df$hp2 = df[["hp"]]^2
  
  # Data for compboost:
  X.hp = as.matrix(df[["hp"]], ncol = 1)
  
  # Use wt as linear baselearner with intercept:
  X.wt = as.matrix(df[["wt"]], ncol = 1)
  X.wt = cbind(1, X.wt)
  
  # Prepare compboost:
  # ------------------
  
  # Create new linear baselearner of hp and wt:
  linear.factory.hp = PolynomialFactory$new(X.hp, "hp", 1)
  linear.factory.wt = PolynomialFactory$new(X.wt, "wt", 1)
  
  # Create new quadratic baselearner of hp:
  quadratic.factory.hp = PolynomialFactory$new(X.hp, "hp", 2)
  
  # Create new polynomial baselearner to the power of 5:
  pow5.factory.hp = PolynomialFactory$new(X.hp, "hp", 5)
  
  # Create new factory list:
  factory.list = FactoryList$new()
  
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
})