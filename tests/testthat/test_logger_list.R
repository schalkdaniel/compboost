context("LoggerList works")

test_that("register and delete of logger entries works", {
  
  expect_silent({ log.iterations = LoggerIteration$new(" iterations", TRUE, 20) })
  expect_silent({ log.time       = LoggerTime$new("time.microseconds", FALSE, 500, "microseconds") })
  expect_error({ LoggerTime$new(FALSE, 300, "hours") })
  
  expect_silent({ logger.list = LoggerList$new() })
  
  expect_silent(logger.list$registerLogger(log.iterations))
  expect_silent(logger.list$registerLogger(log.time))
  
  expect_equal(logger.list$getNumberOfRegisteredLogger(), 2)
  expect_equal(logger.list$getNamesOfRegisteredLogger(), c(" iterations", "time.microseconds"))
  
  expect_silent(logger.list$clearRegisteredLogger())
  
  expect_equal(logger.list$getNumberOfRegisteredLogger(), 0)
  expect_equal(logger.list$getNamesOfRegisteredLogger(), character(0L))  
})
