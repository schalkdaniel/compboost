#' @title Get C++ example script to define a custom cpp logger
#'
#' @description This function can be used to print the trace of the parameters
#'   of a trained compboost object.
#'
#' @param example [\code{character(1)}] \cr
#'   Character value indicating if an example for the base-learner or for the
#'   loss should be returned. The values has to be one of \code{blearner} or \code{loss}.
#' @param silent [\code{logical(1)}] \cr
#'   Logical value indicating if the example code should be printed to the screen or not. 
#' @return 
#'   This function returns a character vector that can be compiled using
#'   \code{Rcpp::sourceCpp(code = getCustomCppExample())} to define a new 
#'   custom cpp logger.
#'   
#' @export
getCustomCppExample = function (example = "blearner", silent = FALSE)
{
  if (! example %in% c("blearner", "loss")) {
    warning("'example' has to be 'blearner' or 'loss'. Setting example to 'blearner'.")
    example = "blearner"
  }
  
  code_blearner = "
  // Example for a linear baselearner:
  // ---------------------------------
  
  // [[Rcpp::depends(RcppArmadillo)]]
  #include <RcppArmadillo.h>
  
  typedef arma::mat (*instantiateDataFunPtr) (const arma::mat& X);
  typedef arma::mat (*trainFunPtr) (const arma::vec& y, const arma::mat& X);
  typedef arma::mat (*predictFunPtr) (const arma::mat& newdata, const arma::mat& parameter);
  
  
  // instantiateDataFun:
  // -------------------
  
  arma::mat instantiateDataFun (const arma::mat& X)
  {
  return X;
  }
  
  // trainFun:
  // -------------------
  
  arma::mat trainFun (const arma::vec& y, const arma::mat& X)
  {
  return arma::solve(X, y);
  }
  
  // predictFun:
  // -------------------
  
  arma::mat predictFun (const arma::mat& newdata, const arma::mat& parameter)
  {
  return newdata * parameter;
  }
  
  
  // Setter function:
  // ------------------
  
  // Now here we wrap the function to an XPtr. This one stores the pointer
  // to the function and can be used as parameter for the BaselearnerCustomCppFactory.
  
  // Note that we don't have to export the upper functions since we are just
  // interested in the pointer of the functions.
  
  // [[Rcpp::export]]
  Rcpp::XPtr<instantiateDataFunPtr> dataFunSetter ()
  {
  return Rcpp::XPtr<instantiateDataFunPtr> (new instantiateDataFunPtr (&instantiateDataFun));
  }
  
  // [[Rcpp::export]]
  Rcpp::XPtr<trainFunPtr> trainFunSetter ()
  {
  return Rcpp::XPtr<trainFunPtr> (new trainFunPtr (&trainFun));
  }
  
  // [[Rcpp::export]]
  Rcpp::XPtr<predictFunPtr> predictFunSetter ()
  {
  return Rcpp::XPtr<predictFunPtr> (new predictFunPtr (&predictFun));
  }
  "
  
  code_loss = "
  // Example for a quadratic loss:
  // -----------------------------  
  // [[Rcpp::depends(RcppArmadillo)]]
  #include <RcppArmadillo.h>
  
  typedef arma::vec (*lossFunPtr) (const arma::vec& true_value, const arma::vec& prediction);
  typedef arma::vec (*gradFunPtr) (const arma::vec& true_value, const arma::vec& prediction);
  typedef double (*constInitFunPtr) (const arma::vec& true_value);
  
  
  // Loss function:
  // -------------------
  
  arma::vec lossFun (const arma::vec& true_value, const arma::vec& prediction)
  {
  return arma::pow(true_value - prediction, 2) / 2;
  }
  
  // Gradient:
  // -------------------
  
  arma::vec gradFun (const arma::vec& true_value, const arma::vec& prediction)
  {
  return prediction - true_value;
  }
  
  // Constant Initializer:
  // -----------------------
  
  double constInitFun (const arma::vec& true_value)
  {
  return arma::mean(true_value);
  }
  
  
  // Setter function:
  // ------------------
  
  // Now wrap the function to an XPtr. This one stores the pointer
  // to the function and can be used as parameter for the BaselearnerCustomCppFactory.
  
  // Note that it isn't necessary to export the upper functions since we are
  // interested in exporting the pointer not the function.
  
  // [[Rcpp::export]]
  Rcpp::XPtr<lossFunPtr> lossFunSetter ()
  {
  return Rcpp::XPtr<lossFunPtr> (new lossFunPtr (&lossFun));
  }
  
  // [[Rcpp::export]]
  Rcpp::XPtr<gradFunPtr> gradFunSetter ()
  {
  return Rcpp::XPtr<gradFunPtr> (new gradFunPtr (&gradFun));
  }
  
  // [[Rcpp::export]]
  Rcpp::XPtr<constInitFunPtr> constInitFunSetter ()
  {
  return Rcpp::XPtr<constInitFunPtr> (new constInitFunPtr (&constInitFun));
  }
  "
  
  if (example == "blearner") {
    code = code_blearner
  }
  if (example == "loss") {
    code = code_loss
  }
  
  if (! silent) {
    cat(code)
  }
  
  return(invisible(code))
}