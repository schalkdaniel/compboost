// ========================================================================== //
//                                 ___.                          __           //
//        ____  ____   _____ ______\_ |__   ____   ____  _______/  |_         //
//      _/ ___\/  _ \ /     \\____ \| __ \ /  _ \ /  _ \/  ___/\   __\        //
//      \  \__(  <_> )  Y Y  \  |_> > \_\ (  <_> |  <_> )___ \  |  |          //
//       \___  >____/|__|_|  /   __/|___  /\____/ \____/____  > |__|          //
//           \/            \/|__|       \/                  \/                //
//                                                                            //
// ========================================================================== //
//
// Compboost is free software: you can redistribute it and/or modify
// it under the terms of the MIT License.
// Compboost is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// MIT License for more details. You should have received a copy of
// the MIT License along with compboost.
//
// =========================================================================== #

// Doxygen index page:

/*! \mainpage Framework for COMPonentwise BOOSTing
 *
 * \section intro_sec Introduction
 *
 * This manual should give an overview about the structure and functionality
 * of the `compboost` `C++` classes. To get an insight into the underlying
 * theory check out the `compboost` vignettes:
 *
 *
 * \section install_sec Installation
 *
 * Basically, the `C++` code can be exported and be used within any language.
 * The only restriction is to exclude the
 * <a href="https://cran.r-project.org/web/packages/Rcpp/vignettes/Rcpp-introduction.pdf">`Rcpp`</a>
 * specific parts which includes some `Rcpp::Rcout` printer and the custom
 * classes which requires `Rcpp::Function` or external pointer of `R` as well
 * as  the <a href="https://cran.r-project.org/web/packages/RcppArmadillo/vignettes/RcppArmadillo-intro.pdf">`RcppArmadillo`</a>
 * package. To get <a href="http://arma.sourceforge.net">`Armadillo`</a> run
 * independent of `Rcpp` one has to link the library manually.
 *
 * As it can already be suspected, the main intend is to use this package
 * within `R`. This is achived by wrapping the pure `C++` classes by another
 * `C++` wrapper which are then exported as `S4` class using the
 * <a href="https://cran.r-project.org/web/packages/Rcpp/vignettes/Rcpp-modules.pdf">Rcpp
 * modules</a>. So the easiest way of using `compboost` is to install the
 * `R` package:
 *
 * ```
 * devtools::install_github("schalkdaniel/compboost")
 * ```
 *
 *
 */

#ifndef COMPBOOST_H_
#define COMPBOOST_H_

#include <memory>

#include "baselearner_track.h"
#include "optimizer.h"
#include "loss.h"
#include "loggerlist.h"
#include "response.h"

namespace cboost {

// Main class:

class Compboost
{

private:

  std::vector<double> risk;

  // Expand learning_rate to vector:
  double learning_rate;

  bool stop_if_all_stopper_fulfilled;
  bool model_is_trained = false;

  unsigned int actual_iteration;

  std::shared_ptr<response::Response> sh_ptr_response;
  blearnertrack::BaselearnerTrack blearner_track;
  optimizer::Optimizer* used_optimizer;
  loss::Loss* used_loss;
  blearnerlist::BaselearnerFactoryList used_baselearner_list;

  // Vector of loggerlists, needed if one want to continue training:
  std::map<std::string, loggerlist::LoggerList*> logger_map;

public:

  Compboost ();

  Compboost (std::shared_ptr<response::Response>, const double&, const bool&, optimizer::Optimizer*, loss::Loss*,
    loggerlist::LoggerList*, blearnerlist::BaselearnerFactoryList);

  // Basic train function used by trainCompbost and continueTraining:
  void train (const unsigned int&, loggerlist::LoggerList*);

  // Initial training:
  void trainCompboost (const unsigned int&);

  // Retraining after initial training:
  void continueTraining (loggerlist::LoggerList*, const unsigned int&);

  arma::vec getPrediction (const bool&) const;

  std::map<std::string, arma::mat> getParameter () const;
  std::vector<std::string> getSelectedBaselearner () const;

  std::map<std::string, loggerlist::LoggerList*> getLoggerList () const;
  std::map<std::string, arma::mat> getParameterOfIteration (const unsigned int&) const;

  std::pair<std::vector<std::string>, arma::mat> getParameterMatrix () const;

  arma::vec predict () const;
  arma::vec predict (std::map<std::string, std::shared_ptr<data::Data>>, const bool&) const;

  void setToIteration (const unsigned int&);

  arma::mat getOffset () const;
  std::vector<double> getRiskVector () const;

  void summarizeCompboost () const;

  // Destructor:
  ~Compboost ();
};

} // namespace cboost

#endif // COMPBOOST_H_
