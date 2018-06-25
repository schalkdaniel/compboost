context("LoggerList works")

test_that("register and delete of logger entries works", {
  
  expect_silent({ log.iterations = IterationLogger$new(TRUE, 20) })
  expect_silent({ log.time       = TimeLogger$new(FALSE, 500, "microseconds") })
  expect_error({ TimeLogger$new(FALSE, 300, "hours") })
  
  expect_silent({ logger.list = LoggerList$new() })
  
  expect_silent(logger.list$registerLogger(" iterations", log.iterations))
  expect_silent(logger.list$registerLogger("time.microseconds", log.time))
  
  expect_equal(logger.list$getNumberOfRegisteredLogger(), 2)
  expect_equal(logger.list$getNamesOfRegisteredLogger(), c(" iterations", "time.microseconds"))
  
  expect_silent(logger.list$clearRegisteredLogger())
  
  expect_equal(logger.list$getNumberOfRegisteredLogger(), 0)
  expect_equal(logger.list$getNamesOfRegisteredLogger(), character(0L))
  
})
