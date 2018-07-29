context("BlearnerFactoryList of 'compboost'")

test_that("factory list works", {

  df = mtcars
  df$hp2 = df[["hp"]]^2

  X.hp = as.matrix(df[["hp"]], ncol = 1)
  X.wt = as.matrix(df[["wt"]], ncol = 1)
  X.wt = cbind(1, X.wt)
  
  expect_silent({ data.source.hp = InMemoryData$new(X.hp, "hp") })
  expect_silent({ data.source.wt = InMemoryData$new(X.wt, "wt") })
  
  expect_silent({ data.target1 = InMemoryData$new() })
  expect_silent({ data.target2 = InMemoryData$new() })
  expect_silent({ data.target3 = InMemoryData$new() })
  expect_silent({ data.target4 = InMemoryData$new() })
  expect_silent({ linear.factory.hp = BaselearnerPolynomial$new(data.source.hp, data.target1, 1, FALSE) })
  expect_silent({ linear.factory.wt = BaselearnerPolynomial$new(data.source.wt, data.target2, 1, FALSE) })
  expect_silent({ quadratic.factory.hp = BaselearnerPolynomial$new(data.source.hp, data.target3, 2, FALSE) })
  expect_silent({ pow5.factory.hp = BaselearnerPolynomial$new(data.source.hp, data.target4, 5, FALSE) })
  expect_silent({ factory.list = BlearnerFactoryList$new() })
  expect_silent(factory.list$registerFactory(linear.factory.hp))
  expect_silent(factory.list$registerFactory(linear.factory.wt))
  expect_silent(factory.list$registerFactory(quadratic.factory.hp))
  expect_silent(factory.list$registerFactory(pow5.factory.hp))

  factory.names = c(
    "hp_polynomial_degree_1",
    "wt_polynomial_degree_1x11",
    "wt_polynomial_degree_1x12",
    "hp_polynomial_degree_2",
    "hp_polynomial_degree_5"
  )
  model.frame = cbind(
    X.hp, X.wt, X.hp^2, X.hp^5
  )
  model.frame   = model.frame[, order(factory.names)]
  factory.names = sort(factory.names)

  expect_equal(factory.list$getModelFrame()$colnames, factory.names)
  expect_equal(factory.list$getModelFrame()$model.frame, model.frame)
  expect_equal(factory.list$getNumberOfRegisteredFactories(), 4)

  expect_silent(factory.list$clearRegisteredFactories())

  expect_equal(factory.list$getNumberOfRegisteredFactories(), 0)

})
