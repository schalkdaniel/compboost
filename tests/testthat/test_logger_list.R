context("LoggerList works")

test_that("register and delete of logger entries works", {

  expect_error(LoggerIteration$new())
  expect_error(LoggerInbagRisk$new())
  expect_error(LoggerOobRisk$new())
  expect_error(LoggerTime$new())

  expect_silent({ log_iterations = LoggerIteration$new(" iterations", TRUE, 20) })
  expect_silent({ log_time       = LoggerTime$new("time_microseconds", FALSE, 500, "microseconds") })
  expect_error({ LoggerTime$new(FALSE, 300, "hours") })

  expect_silent({ logger_list = LoggerList$new() })

  expect_silent(logger_list$registerLogger(log_iterations))
  expect_silent(logger_list$registerLogger(log_time))

  expect_equal(logger_list$getNumberOfRegisteredLogger(), 2)
  expect_equal(logger_list$getNamesOfRegisteredLogger(), c(" iterations", "time_microseconds"))

  expect_silent(logger_list$clearRegisteredLogger())

  expect_equal(logger_list$getNumberOfRegisteredLogger(), 0)
  expect_equal(logger_list$getNamesOfRegisteredLogger(), character(0L))
})
