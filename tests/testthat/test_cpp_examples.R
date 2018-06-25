context("Examples to compile custom C++ base-learner and losses")

test_that("Base-learner example works", {

	expect_warning(getCustomCppExample(example = "i_am_not_present", silent = TRUE))
	expect_output(getCustomCppExample())
	expect_silent(getCustomCppExample(example = "blearner", silent = TRUE))
	expect_type(getCustomCppExample(example = "blearner", silent = TRUE), "character")
	expect_silent(Rcpp::sourceCpp(code = getCustomCppExample(example = "blearner", silent = TRUE)))
})

test_that("Loss example works", {
  
  expect_silent(getCustomCppExample(example = "loss", silent = TRUE))
	expect_type(getCustomCppExample(example = "loss", silent = TRUE), "character")
	expect_silent(Rcpp::sourceCpp(code = getCustomCppExample(example = "loss", silent = TRUE)))

})
