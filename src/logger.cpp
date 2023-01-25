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

#include "logger.h"

namespace logger
{

std::chrono::system_clock::time_point stringToChrono (const std::string s)
{
  std::chrono::system_clock::time_point tout;
  std::istringstream in{s};
  in >> parse("%Y-%m-%d %H:%M:%S", tout);

  return tout;
}

std::shared_ptr<logger::Logger> jsonToLogger (const json& j)
{
  std::shared_ptr<logger::Logger> l;

  if (j["Class"] == "LoggerIteration") {
    l = std::make_shared<LoggerIteration>(j);
  }
  if (j["Class"] == "LoggerInbagRisk") {
    l = std::make_shared<LoggerInbagRisk>(j);
  }
  if (j["Class"] == "LoggerOobRisk") {
    l = std::make_shared<LoggerOobRisk>(j);
  }
  if (j["Class"] == "LoggerTime") {
    l = std::make_shared<LoggerTime>(j);
  }
  if (l == nullptr) {
    throw std::logic_error("No known class in JSON");
  }
  return l;
}



// -------------------------------------------------------------------------- //
// Abstract 'Logger' class:
// -------------------------------------------------------------------------- //

Logger::Logger (const bool is_stopper, const std::string logger_type, const std::string logger_id)
  : _logger_type ( logger_type ),
    _logger_id   ( logger_id ),
    _is_stopper  ( is_stopper )
{ }

Logger::Logger (const json& j)
  : _logger_type ( j["_logger_type"].get<std::string>() ),
    _logger_id   ( j["_logger_id"].get<std::string>() ),
    _is_stopper  ( j["_is_stopper"].get<bool>() )
{ }

void Logger::setIsStopper (const bool is_stopper) { _is_stopper = is_stopper; }

std::string Logger::getLoggerId   () const { return _logger_id; }
std::string Logger::getLoggerType () const { return _logger_type; }
bool        Logger::isStopper     () const { return _is_stopper; }

json Logger::baseToJson (const std::string cln) const
{
  json j = {
    {"Class", cln},
    {"_logger_type", _logger_type},
    {"_logger_id",   _logger_id},
    {"_is_stopper",  _is_stopper}
  };
  return j;
}


Logger::~Logger () {}


// -------------------------------------------------------------------------- //
// Logger implementations:
// -------------------------------------------------------------------------- //

// LoggerIteration:
// -----------------------

/**
 * \brief Default constructor of class `LoggerIteration`
 *
 * Sets the private member `max_iteration` and the tag if the logger should be
 * used as stopper.
 *
 * \param logger_id0 `std::string` unique identifier of the logger
 * \param is_a_stopper `bool` specify if the logger should be used as stopper
 * \param max_iterations `unsigned int` sets value of the stopping criteria
 *
 */

LoggerIteration::LoggerIteration (const std::string logger_id, const bool is_stopper,
  const unsigned int max_iterations)
  : Logger::Logger  ( is_stopper, "iteration", logger_id),
    _max_iterations ( max_iterations )
{ }

LoggerIteration::LoggerIteration (const json& j)
  : Logger::Logger (j),
    _max_iterations ( j["_max_iterations"].get<unsigned int>() ),
    _iterations     ( j["_iterations"].get<std::vector<unsigned int>>() )
{ }

/**
 * \brief Log current step of compboost iteration of class `LoggerIteration`
 *
 * This function loggs the current iteration.
 *
 * \param current_iteration `unsigned int` of current iteration
 * \param response `arma::vec` of the given response used for training
 * \param prediction `arma::vec` actual prediction of the boosting model at
 *   iteration `current_iteration`
 * \param sh_ptr_blearner `Baselearner*` pointer to the selected baselearner in
 *   iteration `current_iteration`
 * \param offset `double` of the overall offset of the training
 * \param learning_rate `double` lerning rate of the `current_iteration`
 *
 */
void LoggerIteration::logStep (const unsigned int current_iteration, const std::shared_ptr<response::Response>& sh_ptr_response,
  const std::shared_ptr<blearner::Baselearner>& sh_ptr_blearner, const double learning_rate, const double step_size,
  const std::shared_ptr<optimizer::Optimizer>& sh_ptr_optimizer, const std::shared_ptr<blearnerlist::BaselearnerFactoryList>& sh_ptr_factory_list)
{
  _iterations.push_back(current_iteration);
}

/**
 * \brief Stop criteria is fulfilled if the current iteration exceed `max_iteration`
 *
 *
 *
 * \returns `bool` which tells if the stopping criteria is reached or not
 *   (if the logger isn't a stopper then this is always false)
 */

bool LoggerIteration::reachedStopCriteria ()
{
  bool stop_criteria_is_reached = false;

  if (_is_stopper) {
    if (_max_iterations <= _iterations.back()) {
      stop_criteria_is_reached = true;
    }
  }
  return stop_criteria_is_reached;
}

/**
 * \brief Return the data stored within the iteration logger
 *
 * This function returns the logged integer. An issue here is, that the later
 * transformation of all logged data to an `arma::mat` requires `arma::vec` as
 * return value. Therefore the std integer vector is transforemd to an
 * `arma::vec`. We know that this isn't very memory friendly, but the
 * `arma::mat` we use later can just have one type.
 *
 * \return `arma::vec` of iterations.
 */
arma::vec LoggerIteration::getLoggedData () const
{
  // Cast integer vector to double:
  std::vector<double> iterations_double (_iterations.begin(), _iterations.end());

  arma::vec out (iterations_double);
  return out;
}

/**
 * \brief Clear the logger data
 *
 * This is an important thing which is called every time in front of retraining
 * the model. If we don't clear the data, the new iterations are just pasted at
 * the end of the existing vectors which couses some troubles.
 */
void LoggerIteration::clearLoggerData ()
{
  _iterations.clear();
}

/**
 * \brief Print status of current iteration into the console
 *
 * The string which is created in this functions must have exactly the same
 * length as the string from `initializeLoggerPrinter()`. Those strings are
 * printed line by line.
 *
 * \returns `std::string` which includes the log of the current iteration
 */
std::string LoggerIteration::printLoggerStatus () const
{
  std::string max_iters = std::to_string(_max_iterations);
  std::stringstream ss;
  ss << std::setw(2 * max_iters.size() + 1) << std::to_string(_iterations.back()) + "/" + max_iters;

  return ss.str();
}

void LoggerIteration::updateMaxIterations (const unsigned int& new_max_iter)
{
  _max_iterations = new_max_iter;
}

json LoggerIteration::toJson () const
{
  json j = Logger::baseToJson("LoggerIteration");
  j["_max_iterations"] = _max_iterations;
  j["_iterations"] = _iterations;

  return j;
}


// InbagRisk:
// -----------------------

/**
 * \brief Default constructor of class `LoggerInbagRisk`
 *
 * \param logger_id `std::string` unique identifier of the logger
 * \param is_stopper `bool` specify if the logger should be used as stopper
 * \param sh_ptr_loss `Loss*` used loss to calculate the empirical risk (this
 *   can differ from the one used while training the model)
 * \param eps_for_break `double` sets value of the stopping criteria`
 */

LoggerInbagRisk::LoggerInbagRisk (const std::string logger_id, const bool is_stopper, const std::shared_ptr<loss::Loss> sh_ptr_loss,
  const double eps_for_break, const unsigned int patience)
  :  Logger::Logger  ( is_stopper, "inbag_risk", logger_id),
    _sh_ptr_loss     ( sh_ptr_loss ),
    _eps_for_break   ( eps_for_break ),
    _patience        ( patience )
{ }

LoggerInbagRisk::LoggerInbagRisk (const json& j)
  : Logger::Logger  ( j ),
    _sh_ptr_loss     ( loss::jsonToLoss(j["_sh_ptr_loss"]) ),
    _inbag_risk      ( j["_inbag_risk"].get<std::vector<double>>() ),
    _eps_for_break   ( j["_eps_for_break"].get<double>() ),
    _patience        ( j["_patience"].get<unsigned int>() ),
    _count_patience  ( j["_count_patience"].get<unsigned int>() )
{ }

/**
 * \brief Log current step of compboost iteration for class `LoggerInbagRisk`
 *
 * This logger computes the risk for the given training data
 * \f$\mathcal{D}_\mathrm{train} = \{(x_i,\ y_i)\ |\ i \in \{1, \dots, n\}\}\f$
 * and stores it into a vector. The empirical risk \f$\mathcal{R}\f$ for
 * iteration \f$m\f$ is calculated by:
 * \f[
 *   \mathcal{R}_\mathrm{emp}^{[m]} = \frac{1}{|\mathcal{D}_\mathrm{train}|}\sum\limits_{(x,y) \in \mathcal{D}_\mathrm{train}} L(y, \hat{f}^{[m]}(x))
 * \f]
 *
 * **Note:**
 *   - If \f$m=0\f$ than \f$\hat{f}\f$ is just the offset.
 *   - The implementation to calculate \f$\mathcal{R}_\mathrm{emp}^{[m]}\f$ is
 *     done in two steps:
 *        1. Calculate vector `risk_temp` of losses for every observation for
 *           given response \f$y^{(i)}\f$ and prediction \f$\hat{f}^{[m]}(x^{(i)})\f$.
 *        2. Average over `risk_temp`.
 *
 *    This procedure ensures, that it is possible to e.g. use the AUC or any
 *    arbitrary performance measure for risk logging. This gives just one
 *    value for `risk_temp` and therefore the average equals the loss
 *    function. If this is just a value (like for the AUC) then the value is
 *    returned.
 *
 * \param current_iteration `unsigned int` of current iteration
 * \param sh_ptr_response `std::shared_ptr<response::Response>` of the given response used for training
 * \param sh_ptr_blearner `std::shared_ptr<Baselearner>` pointer to the selected baselearner in
 *   iteration `current_iteration`
 * \param learning_rate `double` lerning rate of the `current_iteration`
 * \param step_size `double` step size of the iteration
 * \param sh_ptr_optimizer `std::shared_ptr<optimizer::Optimizer>` optimizer used to find the best base-learner
 *
 */
void LoggerInbagRisk::logStep (const unsigned int current_iteration, const std::shared_ptr<response::Response>& sh_ptr_response,
  const std::shared_ptr<blearner::Baselearner>& sh_ptr_blearner, const double learning_rate, const double step_size,
  const std::shared_ptr<optimizer::Optimizer>& sh_ptr_optimizer, const std::shared_ptr<blearnerlist::BaselearnerFactoryList>& sh_ptr_factory_list)
{
  double temp_risk = sh_ptr_response->calculateEmpiricalRisk(_sh_ptr_loss);

  _inbag_risk.push_back(temp_risk);
}

/**
 * \brief Stop criteria is fulfilled if the relative improvement falls below `eps_for_break`
 *
 * The stopping criteria is fulfilled, if the relative improvement at the
 * current iteration \f$m\f$ \f$\varepsilon^{[m]}\f$ falls under a fixed boundary
 * \f$\varepsilon\f$. Where the relative improvement is defined by
 * \f[
 *   \varepsilon^{[m]} = \frac{\mathcal{R}_\mathrm{emp}^{[m-1]} - \mathcal{R}_\mathrm{emp}^{[m]}}{\mathcal{R}_\mathrm{emp}^{[m-1]}}.
 * \f]
 *
 * The logger stops the algorithm if \f$\varepsilon^{[m]} \leq \varepsilon\f$.
 *
 * \returns `bool` which tells if the stopping criteria is reached or not
 *   (if the logger isn't a stopper then this is always false)
 */
bool LoggerInbagRisk::reachedStopCriteria ()
{
  bool stop_criteria_is_reached = false;

  if (_is_stopper) {
    if (_inbag_risk.size() > 1) {
      // We need to subtract -2 and -1 since c++ start counting by 0 while size
      // returns the actual number of elements, so if just one element exists
      // size returns 1 but we want to access 0:
      double inbag_eps = _inbag_risk[_inbag_risk.size() - 2] - _inbag_risk[_inbag_risk.size() - 1];
      inbag_eps = inbag_eps / _inbag_risk[_inbag_risk.size() - 2];

      if (inbag_eps <= _eps_for_break) {
        _count_patience += 1;
      } else {
        _count_patience = 0;
      }
      if (_count_patience == _patience) { stop_criteria_is_reached = true; }
    }
  }
  return stop_criteria_is_reached;
}

/**
 * \brief Return the data stored within the inbag risk logger
 *
 * This function returns the logged inbag risk.
 *
 * \return `arma::vec` of inbag risk
 */
arma::vec LoggerInbagRisk::getLoggedData () const
{
  arma::vec out (_inbag_risk);
  return out;
}

/**
 * \brief Clear the logger data
 *
 * This is an important thing which is called every time in front of retraining
 * the model. If we don't clear the data, the new iterations are just pasted at
 * the end of the existing vectors which couses some troubles.
 */
void LoggerInbagRisk::clearLoggerData ()
{
  _inbag_risk.clear();
}

/**
 * \brief Print status of current iteration into the console
 *
 * The string which is created in this functions must have exactly the same
 * length as the string from `initializeLoggerPrinter()`. Those strings are
 * printed line by line.
 *
 * \returns `std::string` which includes the log of the current iteration
 */

std::string LoggerInbagRisk::printLoggerStatus () const
{
  std::stringstream ss;
  ss << Logger::getLoggerId() << " = " << std::setprecision(2) << _inbag_risk.back();

  return ss.str();
}

json LoggerInbagRisk::toJson () const
{
  json j = Logger::baseToJson("LoggerInbagRisk");
  j["_sh_ptr_loss"] = _sh_ptr_loss->toJson();
  j["_inbag_risk"] = _inbag_risk;
  j["_eps_for_break"] = _eps_for_break;
  j["_patience"] = _patience;
  j["_count_patience"] = _count_patience;

  return j;
}

// OobRisk:
// -----------------------

/**
 * \brief Default constructor of `LoggerOobRisk`
 *
 * \param logger_id `std::string` unique identifier of the logger
 * \param is_stopper `bool` specify if the logger should be used as stopper
 * \param sh_ptr_loss `Loss*` used loss to calculate the empirical risk (this
 *   can differ from the one used while training the model)
 * \param eps_for_break `double` sets value of the stopping criteria`
 */

LoggerOobRisk::LoggerOobRisk (const std::string logger_id, const bool is_stopper,
  const std::shared_ptr<loss::Loss> sh_ptr_loss, const double eps_for_break,
  const unsigned int patience, const std::map<std::string, std::shared_ptr<data::Data>> oob_data_map,
  std::shared_ptr<response::Response> oob_response)
  : Logger::Logger       ( is_stopper, "inbag_risk", logger_id),
    _sh_ptr_loss         ( sh_ptr_loss ),
    _eps_for_break       ( eps_for_break ),
    _patience            ( patience ),
    _oob_data_map        ( oob_data_map ),
    _sh_ptr_oob_response ( oob_response )
{ }

LoggerOobRisk::LoggerOobRisk (const json& j)
  : Logger::Logger  ( j ),
    _sh_ptr_loss    ( loss::jsonToLoss(j["_sh_ptr_loss"]) ),
    _oob_risk       ( j["_oob_risk"].get<std::vector<double>>() ),
    _eps_for_break  ( j["_eps_for_break"].get<double>() ),
    _patience       ( j["_patience"].get<unsigned int>() ),
    _count_patience ( j["_count_patience"].get<unsigned int>() ),
    _oob_prediction ( saver::jsonToArmaMat(j["_oob_prediction"]) ),
    _oob_data_map   ( data::jsonToDataMap(j["_oob_data_map"]) ),
    _oob_data_map_inst   ( data::jsonToDataMap(j["_oob_data_map_inst"]) ),
    _sh_ptr_oob_response ( response::jsonToResponse(j["_sh_ptr_oob_response"]) )
{ }

/**
 * \brief Log current step of compboost iteration for class `LoggerOobRisk`
 *
 * This logger computes the risk for a given new dataset
 * \f$\mathcal{D}_\mathrm{oob} = \{(x_i,\ y_i)\ |\ i \in I_\mathrm{oob}\}\f$
 * and stores it into a vector. The OOB risk \f$\mathcal{R}_\mathrm{oob}\f$ for
 * iteration \f$m\f$ is calculated by:
 * \f[
 *   \mathcal{R}_\mathrm{oob}^{[m]} = \frac{1}{|\mathcal{D}_\mathrm{oob}|}\sum\limits_{(x,y) \in \mathcal{D}_\mathrm{oob}}
 *   L(y, \hat{f}^{[m]}(x))
 * \f]
 *
 * **Note:**
 *   - If \f$m=0\f$ than \f$\hat{f}\f$ is just the offset.
 *   - The implementation to calculate \f$\mathcal{R}_\mathrm{oob}^{[m]}\f$ is
 *     done in two steps:
 *        1. Calculate vector `risk_temp` of losses for every observation for
 *           given response \f$y^{(i)}\f$ and prediction \f$\hat{f}^{[m]}(x^{(i)})\f$.
 *        2. Average over `risk_temp`.
 *
 *    This procedure ensures, that it is possible to e.g. use the AUC or any
 *    arbitrary performance measure for risk logging. This gives just one
 *    value for `risk_temp` and therefore the average equals the loss
 *    function. If this is just a value (like for the AUC) then the value is
 *    returned.
 *
 * \param current_iteration `unsigned int` of current iteration
 * \param sh_ptr_response `std::shared_ptr<response::Response>` of the given response used for training
 * \param sh_ptr_blearner `std::shared_ptr<Baselearner>` pointer to the selected baselearner in
 *   iteration `current_iteration`
 * \param learning_rate `double` lerning rate of the `current_iteration`
 * \param step_size `double` step size of the iteration
 * \param sh_ptr_optimizer `std::shared_ptr<optimizer::Optimizer>` optimizer used to find the best base-learner
 *
 */
void LoggerOobRisk::logStep (const unsigned int current_iteration, const std::shared_ptr<response::Response>& sh_ptr_response,
  const std::shared_ptr<blearner::Baselearner>& sh_ptr_blearner, const double learning_rate, const double step_size,
  const std::shared_ptr<optimizer::Optimizer>& sh_ptr_optimizer, const std::shared_ptr<blearnerlist::BaselearnerFactoryList>& sh_ptr_factory_list)
{

  if (current_iteration == 1) {
    _sh_ptr_oob_response->constantInitialization(_sh_ptr_loss);
    _sh_ptr_oob_response->initializePrediction();
  }
  std::string blearner_id = sh_ptr_blearner->getDataIdentifier();

  // Get data of corresponding selected baselearner. E.g. iteration 100 linear
  // baselearner of feature x_7, then get the data of feature x_7:
  std::string factory_id = sh_ptr_blearner->getDataIdentifier() + "_" + sh_ptr_blearner->getBaselearnerType();
  auto it_oob_data_inst = _oob_data_map_inst.find(factory_id);
  arma::mat temp_oob_prediction;

  auto itf = sh_ptr_factory_list->getFactoryMap().find(factory_id);
  if (it_oob_data_inst == _oob_data_map_inst.end()) {
    auto insert = std::pair<std::string, std::shared_ptr<data::Data>>(factory_id, itf->second->instantiateData(_oob_data_map));
    _oob_data_map_inst.insert(insert);
    temp_oob_prediction = sh_ptr_blearner->predict(insert.second);
  } else {
    temp_oob_prediction = sh_ptr_blearner->predict(it_oob_data_inst->second);
  }

  _sh_ptr_oob_response->updatePrediction(sh_ptr_optimizer->calculateUpdate(learning_rate, step_size, temp_oob_prediction,
    _oob_data_map, _sh_ptr_oob_response));

  double temp_risk = _sh_ptr_oob_response->calculateEmpiricalRisk(_sh_ptr_loss);
  _oob_risk.push_back(temp_risk);
}

/**
 * \brief Stop criteria is fulfilled if the relative improvement falls below
 *   `eps_for_break`
 *
 * The stopping criteria is fulfilled, if the relative improvement at the
 * current iteration \f$m\f$ \f$\varepsilon^{[m]}\f$ falls under a fixed boundary
 * \f$\varepsilon\f$. Where the relative improvement is defined by
 * \f[
 *   \varepsilon^{[m]} = \frac{\mathcal{R}_\mathrm{oob}^{[m-1]} - \mathcal{R}_\mathrm{oob}^{[m]}}{\mathcal{R}_\mathrm{oob}^{[m-1]}}.
 * \f]
 *
 * The logger stops the algorithm if \f$\varepsilon^{[m]} \leq \varepsilon\f$
 *
 * \returns `bool` which tells if the stopping criteria is reached or not
 *   (if the logger isn't a stopper then this is always false)
 */
bool LoggerOobRisk::reachedStopCriteria ()
{
  bool stop_criteria_is_reached = false;

  if (_is_stopper) {
    if (_oob_risk.size() > 1) {
      // We need to subtract -2 and -1 since c++ start counting by 0 while size
      // returns the actual number of elements, so if just one element exists
      // size returns 1 but we want to access 0:
      double oob_eps = _oob_risk[_oob_risk.size() - 2] - _oob_risk[_oob_risk.size() - 1];
      oob_eps = oob_eps / _oob_risk[_oob_risk.size() - 2];

      if (oob_eps <= _eps_for_break) {
        _count_patience += 1;
      } else {
        _count_patience = 0;
      }
      if (_count_patience == _patience) { stop_criteria_is_reached = true; }
    }
  }
  return stop_criteria_is_reached;
}

/**
 * \brief Return the data stored within the OOB risk logger
 *
 * This function returns the logged OOB risk.
 *
 * \return `arma::vec` of out of bag risk
 */
arma::vec LoggerOobRisk::getLoggedData () const
{
  arma::vec out (_oob_risk);
  return out;
}

/**
 * \brief Clear the logger data
 *
 * This is an important thing which is called every time in front of retraining
 * the model. If we don't clear the data, the new iterations are just pasted at
 * the end of the existing vectors which couses some troubles.
 */
void LoggerOobRisk::clearLoggerData ()
{
  _oob_risk.clear();
}

/**
 * \brief Print status of current iteration into the console
 *
 * The string which is created in this functions must have exactly the same
 * length as the string from `initializeLoggerPrinter()`. Those strings are
 * printed line by line.
 *
 * \returns `std::string` which includes the log of the current iteration
 */
std::string LoggerOobRisk::printLoggerStatus () const
{
  std::stringstream ss;
  ss << Logger::getLoggerId() << " = " << std::setprecision(2) << _oob_risk.back();

  return ss.str();
}

json LoggerOobRisk::toJson () const
{
  json j = Logger::baseToJson("LoggerOobRisk");
  j["_sh_ptr_loss"] = _sh_ptr_loss->toJson();
  j["_oob_risk"] = _oob_risk;
  j["_eps_for_break"] = _eps_for_break;
  j["_patience"] = _patience;
  j["_count_patience"] = _count_patience;
  j["_oob_prediction"] = saver::armaMatToJson(_oob_prediction);
  j["_oob_data_map"] = data::dataMapToJson(_oob_data_map);
  j["_oob_data_map_inst"] = data::dataMapToJson(_oob_data_map_inst);
  j["_sh_ptr_oob_response"] = _sh_ptr_oob_response->toJson();

  return j;
}



// LoggerTime:
// -----------------------

/**
 * \brief Default constructor of class `LoggerTime`
 *
 * \param logger_id `std::string` unique identifier of the logger
 * \param is_stopper `bool` which specifies if the logger is used as stopper
 * \param max_time `unsigned int` maximal time for training (just used if logger
 *   is a stopper)
 * \param time_unit `std::string` of the unit used for measuring, allowed are
 *   `minutes`, `seconds` and `microseconds`
 */

LoggerTime::LoggerTime (const std::string logger_id, const bool is_stopper,
  const unsigned int max_time, const std::string time_unit)
  : Logger::Logger ( is_stopper, "time", logger_id),
    _max_time      ( max_time ),
    _time_unit     ( time_unit )
{
  // This is necessary to prevent the program from segfolds... whyever???
  // Copied from: http://lists.r-forge.r-project.org/pipermail/rcpp-devel/2012-November/004796.html
  try {
    std::vector<std::string> choices{ "minutes", "seconds", "microseconds" };
    helper::assertChoice(_time_unit, choices);
    // ************************************************************
    // Old check:
    // Puh, that's ugly. :)
    // if (_time_unit != "minutes" ) {
    //   if (_time_unit != "seconds") {
    //     if (_time_unit != "microseconds") {
    //       Rcpp::stop("Time unit has to be one of 'microseconds', 'seconds' or 'minutes'.");
    //     }
    //   }
    // }
    // ************************************************************
  } catch ( std::exception &ex ) {
    forward_exception_to_r( ex );
  } catch (...) {
    ::Rf_error( "c++ exception (unknown reason)" );
  }
}

LoggerTime::LoggerTime (const json& j)
  : Logger::Logger ( j ),
    _init_time     ( stringToChrono(j["_init_time"].get<std::string>()) ),
    _current_time  ( j["_current_time"].get<std::vector<unsigned int>>() ),
    _retrain_drift ( j["_retrain_drift"].get<unsigned int>() ),
    _max_time      ( j["_max_time"].get<unsigned int>() ),
    _time_unit     ( j["_time_unit"].get<std::string>() )
{ }

/**
 * \brief Log current step of compboost iteration for class `LoggerTime`
 *
 * This functions loggs dependent on `time_unit` the elapsed time at the
 * current iteration.
 *
 * \param current_iteration `unsigned int` of current iteration
 * \param sh_ptr_response `std::shared_ptr<response::Response>` of the given response used for training
 * \param sh_ptr_blearner `std::shared_ptr<Baselearner>` pointer to the selected baselearner in
 *   iteration `current_iteration`
 * \param learning_rate `double` lerning rate of the `current_iteration`
 * \param step_size `double` step size of the iteration
 * \param sh_ptr_optimizer `std::shared_ptr<optimizer::Optimizer>` optimizer used to find the best base-learner
 *
 */
void LoggerTime::logStep (const unsigned int current_iteration, const std::shared_ptr<response::Response>& sh_ptr_response,
  const std::shared_ptr<blearner::Baselearner>& sh_ptr_blearner, const double learning_rate, const double step_size,
  const std::shared_ptr<optimizer::Optimizer>& sh_ptr_optimizer, const std::shared_ptr<blearnerlist::BaselearnerFactoryList>& sh_ptr_factory_list)
{
  if (_current_time.size() == 0) {
    _init_time = std::chrono::system_clock::now();
  }
  unsigned int interim_time;
  if (_time_unit == "minutes") {
    interim_time = std::chrono::duration_cast<std::chrono::minutes>(std::chrono::system_clock::now() - _init_time).count();
  }
  if (_time_unit == "seconds") {
    interim_time = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - _init_time).count();
  }
  if (_time_unit == "microseconds") {
    interim_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - _init_time).count();
  }
  _current_time.push_back(interim_time + _retrain_drift);
}

/**
 * \brief Stop criteria is fulfilled if the passed time exceeds `max_time`
 *
 * The stop criteria here is quite simple. For the current iteration \f$m\f$ it
 * is triggered if
 * \f[
 *   \mathrm{current_time}_m > \mathrm{max_time}
 * \f]
 *
 * \returns `bool` which tells if the stopping criteria is reached or not
 *   (if the logger isn't a stopper then this is always false)
 */
bool LoggerTime::reachedStopCriteria ()
{
  bool stop_criteria_is_reached = false;

  if (_is_stopper) {
    if (_current_time.back() >= _max_time) {
      stop_criteria_is_reached = true;
    }
  }
  return stop_criteria_is_reached;
}

/**
 * \brief Return the data stored within the time logger
 *
 * This function returns the logged elapsed time. An issue here is, that the
 * later transformation of all logged data to an `arma::mat` requires
 * `arma::vec` as return value. Therefore the std integer vector is transforemd
 * to an `arma::vec`. We know that this isn't very memory friendly, but the
 * `arma::mat` we use later can just have one type.
 *
 * \return `arma::vec` of elapsed time
 */
arma::vec LoggerTime::getLoggedData () const
{
  // Cast integer vector to double:
  std::vector<double> time_double (_current_time.begin(), _current_time.end());

  arma::vec out (time_double);
  return out;
}

/**
 * \brief Clear the logger data
 *
 * This is an important thing which is called every time in front of retraining
 * the model. If we don't clear the data, the new iterations are just pasted at
 * the end of the existing vectors which couses some troubles.
 */
void LoggerTime::clearLoggerData ()
{
  _current_time.clear();
}

/**
 * \brief Print status of current iteration into the console
 *
 * The string which is created in this functions must have exactly the same
 * length as the string from `initializeLoggerPrinter()`. Those strings are
 * printed line by line.
 *
 * \returns `std::string` which includes the log of the current iteration
 */
std::string LoggerTime::printLoggerStatus () const
{
  std::stringstream ss;
  ss << Logger::getLoggerId() << " = " << std::setprecision(2) << _current_time.back();

  return ss.str();
}

json LoggerTime::toJson () const
{
  std::ostringstream os;
  os << _init_time;

  json j = Logger::baseToJson("LoggerTime");
  j["_init_time"] = os.str();
  j["_current_time"] = _current_time;
  j["_retrain_drift"] = _retrain_drift;
  j["_max_time"] = _max_time;
  j["_time_unit"] = _time_unit;

  return j;
}

void LoggerTime::reInitializeTime ()
{
  _init_time = std::chrono::system_clock::now();
  _retrain_drift += _current_time.back();
}

} // namespace logger
