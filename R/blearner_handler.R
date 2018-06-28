.handleRcpp_PolynomialBlearner = function (degree = 1, intercept = TRUE, ...) {

	nuisance = list(...)
	if (length(nuisance) > 0) {
		warning("Following arguments are ignored by the polynomial base-learner: ", paste(names(nuisance), collapse = ", "))
	}
	params = list(degree = degree, intercept = intercept)

	return (params)
}

.handleRcpp_PSplineBlearner = function (degree = 3, n.knots = 20, penalty = 2, differences = 2, ...) {

	nuisance = list(...)
	if (length(nuisance) > 0) {
		warning("Following arguments are ignored by the spline base-learner: ", paste(names(nuisance), collapse = ", "))
	}
	params = list(degree = degree, n.knots = n.knots, penalty = penalty, differences = differences)

	return (params)
}

.handleRcpp_CustomBlearner = function (instantiate.fun, train.fun, predict.fun, param.fun, ...) {

	nuisance = list(...)
	if (length(nuisance) > 0) {
		warning("Following arguments are ignored by the custom base-learner: ", paste(names(nuisance), collapse = ", "))
	}
	params = list(instantiate.fun = instantiate.fun, train.fun = train.fun, predict.fun = predict.fun, param.fun = param.fun)

	return (params)
}

.handleRcpp_CustomCppBlearner = function (instantiate.ptr, train.ptr, predict.ptr, ...) {

	nuisance = list(...)
	if (length(nuisance) > 0) {
		warning("Following arguments are ignored by the custom cpp base-learner: ", paste(names(nuisance), collapse = ", "))
	}
	params = list(instantiate.ptr = instantiate.ptr, train.ptr = train.ptr, predict.ptr = predict.ptr)

	return (params)
}
