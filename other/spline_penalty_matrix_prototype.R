# ============================================================================ #
#                                                                              #
#                       Prototype to create Penalty Matrix                     #
#                                                                              #
# ============================================================================ #

cpp.fun1 = "
arma::vec test1 (const arma::vec& myvec, const arma::mat& mymat)
{
  return arma::solve(mymat, myvec);
}
"

cpp.fun2 = "
arma::vec test2 (const arma::vec& myvec, const arma::mat& mymat)
{
  return arma::inv(mymat.t() * mymat) * mymat.t() * myvec;
}
"

Rcpp::cppFunction(cpp.fun1, depends = "RcppArmadillo", plugins = "cpp11")
Rcpp::cppFunction(cpp.fun2, depends = "RcppArmadillo", plugins = "cpp11")

test3 = function (y, X)
{
  return (solve(t(X) %*% X) %*% t(X) %*% y)
}

mydata = na.omit(hflights::hflights)

y = mydata$ArrDelay

X = cbind(mydata$DepDelay, mydata$AirTime, mydata$Distance, mydata$TaxiIn)



test1(y, X)
test2(y, X)
test3(y, X)

microbenchmark::microbenchmark(
  "c++1" = test1(y,X),
  "c++2" = test2(y,X),
  "r"    = test3(y,X)
)






cpp.fun.sparse = "
arma::sp_mat testSparse (const arma::mat& mymat, const arma::vec& myvec, 
  const unsigned int& differences)
{
  arma::sp_mat X(mymat);

  arma::sp_mat newX = arma::join_cols(X, X);
  newX = X.t() * X;

  unsigned int d = myvec.size();

  arma::sp_mat diffs(0, d);
  for (unsigned int i = 0; i < d-1; i++) {
    arma::sp_mat insert(1, d);
    insert[i] = -1;
    insert[i + 1] = 1;
    diffs = join_cols(diffs, insert);
  }

  for (unsigned int k = 0; k < differences - 1; k++) {
    arma::sp_mat sparse_temp = diffs(arma::span(1, diffs.n_rows - 1), arma::span(1, diffs.n_cols - 1));
    diffs = sparse_temp * diffs;
  }

  arma::mat out = arma::spsolve(X, myvec, \"lapack\");

  return diffs;
}
"

src.get.K = "
arma::sp_mat penaltyMat (const unsigned int& n, const unsigned int& differences)
{
  // Create frame for sparse difference matrix:
  arma::sp_mat diffs(0, n);
  for (unsigned int i = 0; i < n-1; i++) {
    arma::sp_mat insert(1, n);
    insert[i] = -1;
    insert[i + 1] = 1;
    diffs = join_cols(diffs, insert);
  }

  // Calculate the difference matrix for higher orders:
  if (differences > 1) {
    arma::sp_mat diffs_reduced = diffs;
    for (unsigned int k = 0; k < differences - 1; k++) {
      diffs_reduced = diffs_reduced(arma::span(1, diffs_reduced.n_rows - 1), arma::span(1, diffs_reduced.n_cols - 1));
      diffs = diffs_reduced * diffs;
    }
  }

  arma::sp_mat K = diffs.t() * diffs;

  return K;
}
"

getDiffK = function (n, d)
{
  D = diff(diag(n), differences = d)
  
  return (t(D) %*% D)
}

Rcpp::cppFunction(cpp.fun.sparse, depends = "RcppArmadillo", plugins = "cpp11")
Rcpp::cppFunction(src.get.K, depends = "RcppArmadillo")

# a = testSparse(diag(10), rnorm(10))
# t(a) %*% a
# 


getD1 = function (n)
{
  D = diag(x = - 1, ncol = n, nrow = n - 1)
  D[, -1] = D[, -1] + diag(x = 1, ncol = n - 1, nrow = n - 1)
  return(D)
}
getD = function (n, d)
{
  D = getD1(n = n)
  for (i in seq_len(d - 1)) {
    D = getD1(n = n - i) %*% D
  }
  return(D)
}
getK = function (n, d)
{
  D = getD(n = n, d = d)
  return(t(D) %*% D)
}

getDiffK = function (n, d)
{
  D = diff(diag(n), differences = d)
  return (t(D) %*% D)
}

getK(10, 3)
penaltyMat(10, 3)

microbenchmark::microbenchmark(
  "C++" = penaltyMat(100, 4),
  # "R"   = getK(1000, 4),
  "R fast" = getDiffK(100, 4),
  times = 10L
)

pryr::mem_change(penaltyMat(1000, 4))
pryr::mem_change(getK(1000, 4))

a = penaltyMat(10000, 4)
b = getK(1000, 4)
c = getDiffK(10000, 3)

pryr::object_size(a)
pryr::object_size(b)
