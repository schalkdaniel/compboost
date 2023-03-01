context("BlearnerFactoryList of 'compboost'")

test_that("factory list works", {

  df = mtcars
  df$hp2 = df[["hp"]]^2

  X_hp = as.matrix(df[["hp"]], ncol = 1)
  X_wt = as.matrix(df[["wt"]], ncol = 1)

  expect_silent({ data_source_hp = InMemoryData$new(X_hp, "hp") })
  expect_silent({ data_source_wt = InMemoryData$new(X_wt, "wt") })

  expect_silent({ linear_factory_hp = BaselearnerPolynomial$new(data_source_hp,
    list(degree = 1, intercept = FALSE)) })
  expect_silent({ quadratic_factory_wt = BaselearnerPolynomial$new(data_source_wt,
    list(degree = 2, intercept = FALSE)) })
  expect_silent({ pow5_factory_hp = BaselearnerPolynomial$new(data_source_hp,
    list(degree = 5, intercept = FALSE)) })
  expect_silent({ factory_list = BlearnerFactoryList$new() })
  expect_silent(factory_list$registerFactory(linear_factory_hp))
  expect_silent(factory_list$registerFactory(quadratic_factory_wt))
  expect_silent(factory_list$registerFactory(pow5_factory_hp))

  factory_names = c(
    "hp_poly1",
    "wt_poly2x11",
    "wt_poly2x12",
    "hp_poly5x11",
    "hp_poly5x12",
    "hp_poly5x13",
    "hp_poly5x14",
    "hp_poly5x15"
  )
  model_frame = cbind(
    X_hp, X_wt, X_wt^2, X_hp, X_hp^2, X_hp^3, X_hp^4, X_hp^5
  )
  model_frame   = model_frame[, order(factory_names)]
  factory_names = sort(factory_names)

  expect_equal(factory_list$getModelFrame()$colnames, factory_names)
  expect_equal(factory_list$getModelFrame()$model_frame, model_frame)
  expect_equal(factory_list$getNumberOfRegisteredFactories(), 3)

  expect_silent(factory_list$clearRegisteredFactories())

  expect_equal(factory_list$getNumberOfRegisteredFactories(), 0)
})
