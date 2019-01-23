context("BlearnerFactoryList of 'compboost'")

test_that("factory list works", {

  df = mtcars
  df$hp2 = df[["hp"]]^2

  X_hp = as.matrix(df[["hp"]], ncol = 1)
  X_wt = as.matrix(df[["wt"]], ncol = 1)
  X_wt = cbind(1, X_wt)
  
  expect_silent({ data_source_hp = InMemoryData$new(X_hp, "hp") })
  expect_silent({ data_source_wt = InMemoryData$new(X_wt, "wt") })
  
  expect_silent({ data_target1 = InMemoryData$new() })
  expect_silent({ data_target2 = InMemoryData$new() })
  expect_silent({ data_target3 = InMemoryData$new() })
  expect_silent({ data_target4 = InMemoryData$new() })
  expect_silent({ linear_factory_hp = BaselearnerPolynomial$new(data_source_hp, data_target1, 
    list(degree = 1, intercept = FALSE)) })
  expect_silent({ linear_factory_wt = BaselearnerPolynomial$new(data_source_wt, data_target2, 
    list(degree = 1, intercept = FALSE)) })
  expect_silent({ quadratic_factory_hp = BaselearnerPolynomial$new(data_source_hp, data_target3, 
    list(degree = 2, intercept = FALSE)) })
  expect_silent({ pow5_factory_hp = BaselearnerPolynomial$new(data_source_hp, data_target4, 
    list(degree = 5, intercept = FALSE)) })
  expect_silent({ factory_list = BlearnerFactoryList$new() })
  expect_silent(factory_list$registerFactory(linear_factory_hp))
  expect_silent(factory_list$registerFactory(linear_factory_wt))
  expect_silent(factory_list$registerFactory(quadratic_factory_hp))
  expect_silent(factory_list$registerFactory(pow5_factory_hp))

  factory_names = c(
    "hp_polynomial_degree_1",
    "wt_polynomial_degree_1x11",
    "wt_polynomial_degree_1x12",
    "hp_polynomial_degree_2",
    "hp_polynomial_degree_5"
  )
  model_frame = cbind(
    X_hp, X_wt, X_hp^2, X_hp^5
  )
  model_frame   = model_frame[, order(factory_names)]
  factory_names = sort(factory_names)

  expect_equal(factory_list$getModelFrame()$colnames, factory_names)
  expect_equal(factory_list$getModelFrame()$model_frame, model_frame)
  expect_equal(factory_list$getNumberOfRegisteredFactories(), 4)

  expect_silent(factory_list$clearRegisteredFactories())

  expect_equal(factory_list$getNumberOfRegisteredFactories(), 0)
})
