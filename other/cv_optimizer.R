source = "
arma::mat getSplits (arma::vec& x, unsigned int& folds)
{
	unsigned int lengths = ceil(x.size() / (double) folds);

	arma::mat from_to_mat(folds, 2, arma::fill::zeros);
	unsigned int from = 1;
	unsigned int to   = lengths;

	for (unsigned int i = 0; i < folds; i++) {
		from_to_mat(i, 0) = from;
		from_to_mat(i, 1) = to;

		from = to + 1;
		to   = from + lengths - 1;

		if (x.size() < to) {
			to = x.size();
		}
	}
	return from_to_mat;
}
"
source2 = "
arma::rowvec matrix_locs(arma::mat M, arma::umat locs) {
  
  arma::uvec eids = sub2ind( size(M), locs ); // Obtain Element IDs
  arma::vec v  = M.elem( eids );              // Values of the Elements
  
  return v.t();                               // Transpose to mimic R
}
"
# Rcpp::cppFunction(code = source, depends = "RcppArmadillo", rebuild = TRUE)
Rcpp::cppFunction(code = source2, depends = "RcppArmadillo", rebuild = TRUE)

(M <- matrix(1:9, 3, 3))   # Construct & Display Matrix
(locs <- rbind(c(1, 2),    # List non-continuous locations
               c(3, 1),
               c(2, 3)))

cpp_locs <- locs - 1       # Shift indices from R to C++
(cpp_locs <- t(cpp_locs))  # Transpose matrix for 2 x n form
matrix_locs(M, cpp_locs)

x = 1:1000
folds = 11

getSplits(x, folds)



getSplitsR = function (x, folds) {
	lgts = trunc(length(x) / folds) + 1

	lst = list()
	from = 1
	to = lgts

	for (i in seq_len(folds)) {
		lst[[i]] = c(from = from, to = to)
		from = to + 1
		to   = from + lgts - 1
		if (length(x) < to) {
			to = length(x)			
		}
	}
	return (lst)
}

# lst

# lapply(lst, function (x) { 
# 	return(length(x["from"]:x["to"])) 
# })

microbenchmark::microbenchmark(
	"C++" = getSplits(x, folds),
	"R"   = getSplitsR(x, folds)
)