using namespace Rcpp;
class Uniform {
public:
	Uniform(double min_, double max_) :
	min(min_), max(max_) {}
	NumericVector draw(int n) const {
		RNGScope scope;
		return runif(n, min, max);
	}
	double min, max;
	double degree = 1;
	// Rcpp::List mylist = Rcpp::List::create(Rcpp::Named("degree") = 1,
 //                          Rcpp::Named("intercept") = TRUE);
};

double uniformRange(Uniform* w) {
	return w->max - w->min;
}

double getDefaults () {
	return 1.1;
}

RCPP_MODULE(unif_module) {
	class_<Uniform>("Uniform")
	.constructor<double,double>()
	.field("min", &Uniform::min)
	.field("max", &Uniform::max)
	.field("degree", &Uniform::degree)
	.method("draw", &Uniform::draw)
	.method("range", &uniformRange)
	;
}