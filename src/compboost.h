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

/*! \mainpage Framework for COMPonentwise BOOSTing
 *
 * \section intro_sec Introduction
 *
 * This manual should give an overview about the structure and functionality
 * of the `compboost` `C++` classes.
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
 */

#ifndef COMPBOOST_H_
#define COMPBOOST_H_

#include <memory>
#include <sstream>
#include <fstream>

#include "baselearner_track.h"
#include "optimizer.h"
#include "loss.h"
#include "loggerlist.h"
#include "response.h"
#include "saver.h"

#include "single_include/nlohmann/json.hpp"
using json = nlohmann::json;

typedef std::shared_ptr<data::Data> sdata;
typedef std::map<std::string, sdata> mdata;

namespace cboost {

class Compboost
{

private:
  const double  _learning_rate;
  const bool    _is_global_stopper;

  const std::shared_ptr<response::Response>      _sh_ptr_response;
  const std::shared_ptr<optimizer::Optimizer>    _sh_ptr_optimizer;
  const std::shared_ptr<loss::Loss>              _sh_ptr_loss;
  const std::shared_ptr<loggerlist::LoggerList>  _sh_ptr_loggerlist;

  bool                 _is_trained = false;
  unsigned int         _current_iter;
  std::vector<double>  _risk;

  std::shared_ptr<blearnerlist::BaselearnerFactoryList> _sh_ptr_factory_list;
  blearnertrack::BaselearnerTrack      _blearner_track;

public:
  Compboost (std::shared_ptr<response::Response>, const double&, const bool&, std::shared_ptr<optimizer::Optimizer>, std::shared_ptr<loss::Loss>,
    std::shared_ptr<loggerlist::LoggerList>, std::shared_ptr<blearnerlist::BaselearnerFactoryList>);
  Compboost (const json&, const mdata&, const mdata&);
  Compboost (const json&);
  Compboost (const std::string);

  // Getter/Setter
  arma::vec                                       getPrediction (const bool&)                    const;
  std::map<std::string, arma::mat>                getParameter ()                                const;
  std::vector<std::string>                        getSelectedBaselearner ()                      const;
  std::shared_ptr<loggerlist::LoggerList>         getLoggerList ()                               const;
  std::map<std::string, arma::mat>                getParameterOfIteration (const unsigned int&)  const;
  std::pair<std::vector<std::string>, arma::mat>  getParameterMatrix ()                          const;
  arma::mat                                       getOffset ()                                   const;
  std::vector<double>                             getRiskVector ()                               const;

  // To provide pointer for the modules:
  double                                                getLearningRate ()   const;
  std::shared_ptr<blearnerlist::BaselearnerFactoryList> getBaselearnerList() const;
  std::shared_ptr<optimizer::Optimizer>                 getOptimizer()       const;


  // Other member functions
  void       train              (const unsigned int, const std::shared_ptr<loggerlist::LoggerList>);
  void       trainCompboost     (const unsigned int);
  void       continueTraining   (const unsigned int);
  arma::vec  predict            () const;
  arma::vec  predict            (const std::map<std::string, std::shared_ptr<data::Data>>&, const bool&) const;
  void       setToIteration     (const unsigned int&, const unsigned int&);
  void       summarizeCompboost () const;

  // Save JSON, to load use the respective constructor:
  void saveJson (std::string) const;

  arma::mat predictFactory (const std::string&) const;
  arma::mat predictFactory (const std::string&, const std::map<std::string, std::shared_ptr<data::Data>>&) const;

  std::map<std::string, arma::mat> predictIndividual () const;
  std::map<std::string, arma::mat> predictIndividual (const std::map<std::string, std::shared_ptr<data::Data>>&) const;

  // Destructor:
  ~Compboost ();
};

} // namespace cboost

#endif // COMPBOOST_H_
