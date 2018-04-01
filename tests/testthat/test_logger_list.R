context("LoggerList works")

test_that("register and delete of logger entries works", {
  
  log.iterations = IterationLogger$new(TRUE, 20)
  log.time       = TimeLogger$new(FALSE, 500, "microseconds")
  
  logger.list = LoggerList$new()
  
  logger.list$registerLogger(" iterations", log.iterations)
  logger.list$registerLogger("time.microseconds", log.time)
  
  expect_equal(logger.list$getNumberOfRegisteredLogger(), 2)
  expect_equal(logger.list$getNamesOfRegisteredLogger(), c(" iterations", "time.microseconds"))
  
  logger.list$clearRegisteredLogger()
  
  expect_equal(logger.list$getNumberOfRegisteredLogger(), 0)
  expect_equal(logger.list$getNamesOfRegisteredLogger(), character(0L))
  
})
