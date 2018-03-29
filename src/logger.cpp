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
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// Compboost is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
// You should have received a copy of the GNU General Public License
// along with Compboost. If not, see <http://www.gnu.org/licenses/>.
//
// This file contains:
// -------------------
//
//   Implementation of "Logger" class.
//
// Written by:
// -----------
//
//   Daniel Schalk
//   Institut für Statistik
//   Ludwig-Maximilians-Universität München
//   Ludwigstraße 33
//   D-80539 München
//
//   https://www.compstat.statistik.uni-muenchen.de
//
// ========================================================================== //

#include "logger.h"

namespace logger
{

// -------------------------------------------------------------------------- //
// Abstract 'Logger' class:
// -------------------------------------------------------------------------- //

/**
 * \brief Getter if the logger is used as stopper
 * 
 * \returns `bool` which says `true` if it is a logger, otherwise `false`
 */
bool Logger::GetIfLoggerIsStopper () const
{
  return is_a_stopper;
}

// Destructor:
Logger::~Logger () { }





// -------------------------------------------------------------------------- //
// Logger implementations:
// -------------------------------------------------------------------------- //

// IterationLogger:
// -----------------------

/**
 * \brief Default constructor of class `IterationLogger`
 * 
 * Sets the private member `max_iteration` and the tag if the logger should be
 * used as stopper.
 * 
 * \param is_a_stopper `bool` specify if the logger should be used as stopper
 * \param max_iterations `unsigned int` sets value of the stopping criteria
 * 
 */

IterationLogger::IterationLogger (const bool& is_a_stopper0, 
  const unsigned int& max_iterations) 
  : max_iterations ( max_iterations ) 
{
  is_a_stopper = is_a_stopper0;
};

/**
 * \brief Log current step of compboost iteration of class `IterationLogger`
 * 
 * This function loggs the current iteration. 
 * 
 * \param current_iteration `unsigned int` of current iteration 
 * \param response `arma::vec` of the given response used for training
 * \param prediction `arma::vec` actual prediction of the boosting model at 
 *   iteration `current_iteration`
 * \param used_blearner `Baselearner*` pointer to the selected baselearner in 
 *   iteration `current_iteration`
 * \param offset `double` of the overall offset of the training
 * \param learning_rate `double` lerning rate of the `current_iteration`
 * 
 */

void IterationLogger::logStep (const unsigned int& current_iteration, const arma::vec& response, 
  const arma::vec& prediction, blearner::Baselearner* used_blearner, const double& offset, 
  const double& learning_rate)
{
  iterations.push_back(current_iteration);
}

/**
 * \brief Stop criteria is fulfilled if the current iteration exceed `max_iteration`
 * 
 * 
 * 
 * \returns `bool` which tells if the stopping criteria is reached or not 
 *   (if the logger isn't a stopper then this is always false)
 */

bool IterationLogger::reachedStopCriteria () const
{
  bool stop_criteria_is_reached = false;
  
  if (is_a_stopper) {
    if (max_iterations <= iterations.back()) {
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

arma::vec IterationLogger::getLoggedData () const
{
  // Cast integer vector to double:
  std::vector<double> iterations_double (iterations.begin(), iterations.end());
  
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

void IterationLogger::clearLoggerData ()
{
  iterations.clear();
}

/**
 * \brief Print the head of the trace which is printed to the console
 * 
 * \returns `std::string` which is used to initialize the header of the trace
 */

std::string IterationLogger::initializeLoggerPrinter () const
{
  // 15 characters:
  return "      Iteration";
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

std::string IterationLogger::printLoggerStatus () const
{
  std::stringstream ss;
  ss << std::setw(15) << std::to_string(iterations.back()) + "/" + std::to_string(max_iterations);
  
  return ss.str();
}





// InbagRisk:
// -----------------------

/**
 * \brief Default constructor of class `InbagRiskLogger`
 * 
 * \param is_a_stopper0 `bool` specify if the logger should be used as stopper
 * \param used_loss `Loss*` used loss to calculate the empirical risk (this 
 *   can differ from the one used while training the model)
 * \param eps_for_break `double` sets value of the stopping criteria`
 */

InbagRiskLogger::InbagRiskLogger (const bool& is_a_stopper0, loss::Loss* used_loss, 
  const double& eps_for_break)
  : used_loss ( used_loss ),
    eps_for_break ( eps_for_break )
{
  is_a_stopper = is_a_stopper0;
}

/**
 * \brief Log current step of compboost iteration for class `InbagRiskLogger`
 * 
 *  * This logger computes the risk for the given training data
 * \f$\mathcal{D}_\mathrm{train} = \{(x_i,\ y_i)\ |\ i \in \{1, \dots, n\}\}\f$
 * and stores it into a vector. The risk \f$\mathcal{R}\f$ for iteration \f$m\f$ 
 * is calculated by:
 * \f[
 *   \mathcal{R}^{[m]} = \frac{1}{|\mathcal{D}_\mathrm{train}|}\sum\limits_{(x,y) \in \mathcal{D}_\mathrm{train}} L(y, \hat{f}(x)^{[m]})
 * \f]
 * 
 * **Note:** If \f$m=0\f$ than \hat{f} is just the offset.
 * 
 * \param current_iteration `unsigned int` of current iteration 
 * \param response `arma::vec` of the given response used for training
 * \param prediction `arma::vec` actual prediction of the boosting model at 
 *   iteration `current_iteration`
 * \param used_blearner `Baselearner*` pointer to the selected baselearner in 
 *   iteration `current_iteration`
 * \param offset `double` of the overall offset of the training
 * \param learning_rate `double` lerning rate of the `current_iteration`
 * 
 */

void InbagRiskLogger::logStep (const unsigned int& current_iteration, const arma::vec& response, 
  const arma::vec& prediction, blearner::Baselearner* used_blearner, const double& offset, 
  const double& learning_rate)
{
  double temp_risk = arma::accu(used_loss->definedLoss(response, prediction)) / response.size();
  
  tracked_inbag_risk.push_back(temp_risk);
}

/**
 * \brief Stop criteria is fulfilled if the relative improvement falls below `eps_for_break`
 * 
 * The stopping criteria is fulfilled, if the relative improvement at the 
 * current iteration \f$m\f$ \varepsilon^{[m]} falls under a fixed boundary 
 * \f$\varepsilon\f$. Where the relative improvement is defined by
 * \f[
 *   \varepsilon^{[m]} = \frac{\mathcal{R}_\mathrm{train}^{[m-1]} - \mathcal{R}_\mathrm{train}^{[m]}}{\mathcal{R}_\mathrm{train}^{[m-1]}}.
 * \f]
 * 
 * The logger stops the algorithm if \f$\varepsilon^{[m]} \leq \varepsilon\f$.
 * 
 * \returns `bool` which tells if the stopping criteria is reached or not 
 *   (if the logger isn't a stopper then this is always false)
 */

bool InbagRiskLogger::reachedStopCriteria () const
{
  bool stop_criteria_is_reached = false;
  
  if (is_a_stopper) {
    if (tracked_inbag_risk.size() > 1) {
      double inbag_eps = tracked_inbag_risk[tracked_inbag_risk.size() - 1] - tracked_inbag_risk[tracked_inbag_risk.size()];
      inbag_eps = inbag_eps / tracked_inbag_risk[tracked_inbag_risk.size() - 1];
      
      if (inbag_eps <= eps_for_break) {
        stop_criteria_is_reached = true;
      }
    }
  }
  return stop_criteria_is_reached;
}

/**
 * \brief Return the data stored within the OOB risk logger
 * 
 * This function returns the logged OOB risk.
 * 
 * \return `arma::vec` of elapsed time
 */

arma::vec InbagRiskLogger::getLoggedData () const
{
  arma::vec out (tracked_inbag_risk);
  return out;
}

/**
 * \brief Clear the logger data
 * 
 * This is an important thing which is called every time in front of retraining 
 * the model. If we don't clear the data, the new iterations are just pasted at
 * the end of the existing vectors which couses some troubles. 
 */

void InbagRiskLogger::clearLoggerData ()
{
  tracked_inbag_risk.clear();
}

/**
 * \brief Print the head of the trace which is printed to the console
 * 
 * \returns `std::string` which is used to initialize the header of the trace
 */

std::string InbagRiskLogger::initializeLoggerPrinter () const
{  
  std::stringstream ss;
  ss << std::setw(17) << "Inbag Risk";
  
  return ss.str();
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

std::string InbagRiskLogger::printLoggerStatus () const
{
  std::stringstream ss;
  ss << std::setw(17) << std::fixed << std::setprecision(2) << tracked_inbag_risk.back();
  
  return ss.str();
}





// OobRisk:
// -----------------------


/**
 * \brief Default constructor of `OobRiskLogger`
 * 
 * \param is_a_stopper0 `bool` to set if the logger should be used as stopper
 * \param used_loss `Loss*` which is used to calculate the empirical risk (this 
 *   can differ from the loss used while trining the model)
 * \param eps_for_break `double` sets value of the stopping criteria
 * \param oob_data `std::map<std::string, data::Data*>` the new data
 * \param oob_response `arma::vec` response of the new data
 */

OobRiskLogger::OobRiskLogger (const bool& is_a_stopper0, loss::Loss* used_loss, 
  const double& eps_for_break, std::map<std::string, data::Data*> oob_data, 
  const arma::vec& oob_response)
  : used_loss ( used_loss ),
    eps_for_break ( eps_for_break ),
    oob_data ( oob_data ),
    oob_response ( oob_response )
{
  is_a_stopper = is_a_stopper0;
  
  arma::vec temp (oob_response.size());
  oob_prediction = temp;
}

/**
 * \brief Log current step of compboost iteration for class `OobRiskLogger`
 * 
 * This logger computes the risk for a given new dataset 
 * \f$\mathcal{D}_\mathrm{oob} = \{(x_i,\ y_i)\ |\ i \in I_\mathrm{oob}\}\f$
 * and stores it into a vector. The OOB risk \f$\mathcal{R}_\mathrm{oob}\f$ for 
 * iteration \f$m\f$ is calculated by:
 * \f[
 *   \mathcal{R}_\mathrm{oob}^{[m]} = \frac{1}{|\mathcal{D}_\mathrm{oob}|}\sum\limits_{(x,y) \in \mathcal{D}_\mathrm{oob}} 
 *   L(y, \hat{f}(x)^{[m]})
 * \f]
 * 
 * **Note:** If \f$m=0\f$ than \hat{f} is just the offset.
 * 
 * \param current_iteration `unsigned int` of current iteration 
 * \param response `arma::vec` of the given response used for training
 * \param prediction `arma::vec` actual prediction of the boosting model at 
 *   iteration `current_iteration`
 * \param used_blearner `Baselearner*` pointer to the selected baselearner in 
 *   iteration `current_iteration`
 * \param offset `double` of the overall offset of the training
 * \param learning_rate `double` lerning rate of the `current_iteration`
 * 
 */

void OobRiskLogger::logStep (const unsigned int& current_iteration, const arma::vec& response, 
  const arma::vec& prediction, blearner::Baselearner* used_blearner, const double& offset, 
  const double& learning_rate)
{
  if (current_iteration == 1) {
    oob_prediction.fill(offset);
  }
  
  // Get data of corresponding selected baselearner. E.g. iteration 100 linear 
  // baselearner of feature x_7, then get the data of feature x_7:
  data::Data* oob_blearner_data = oob_data.find(used_blearner->getDataIdentifier())->second;
  
  // Predict this data using the selected baselearner:
  arma::vec temp_oob_prediction = used_blearner->predict(oob_blearner_data);
  
  // Cumulate prediction and shrink by learning rate:
  oob_prediction += learning_rate * temp_oob_prediction;
  
  // Calculate empirical risk:
  double temp_risk = arma::accu(used_loss->definedLoss(oob_response, oob_prediction)) / response.size();
  
  // Track empirical risk:
  tracked_oob_risk.push_back(temp_risk);
}

/**
 * \brief Stop criteria is fulfilled if the relative improvement falls below 
 *   `eps_for_break`
 * 
 * The stopping criteria is fulfilled, if the relative improvement at the 
 * current iteration \f$m\f$ \varepsilon^{[m]} falls under a fixed boundary 
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

bool OobRiskLogger::reachedStopCriteria () const
{
  bool stop_criteria_is_reached = false;
  
  if (is_a_stopper) {
    if (tracked_oob_risk.size() > 1) {
      double oob_eps = tracked_oob_risk[tracked_oob_risk.size() - 1] - tracked_oob_risk[tracked_oob_risk.size()];
      oob_eps = oob_eps / tracked_oob_risk[tracked_oob_risk.size() - 1];
      
      if (oob_eps <= eps_for_break) {
        stop_criteria_is_reached = true;
      }
    }
  }
  return stop_criteria_is_reached;
}

/**
 * \brief Return the data stored within the OOB risk logger
 * 
 * This function returns the logged OOB risk.
 * 
 * \return `arma::vec` of elapsed out of bag risk
 */

arma::vec OobRiskLogger::getLoggedData () const
{
  arma::vec out (tracked_oob_risk);
  return out;
}

/**
 * \brief Clear the logger data
 * 
 * This is an important thing which is called every time in front of retraining 
 * the model. If we don't clear the data, the new iterations are just pasted at
 * the end of the existing vectors which couses some troubles. 
 */
void OobRiskLogger::clearLoggerData ()
{
  tracked_oob_risk.clear();
}

/**
 * \brief Print the head of the trace which is printed to the console
 * 
 * \returns `std::string` which is used to initialize the header of the trace
 */

std::string OobRiskLogger::initializeLoggerPrinter () const
{  
  std::stringstream ss;
  ss << std::setw(17) << "Out of Bag Risk";
  
  return ss.str();
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

std::string OobRiskLogger::printLoggerStatus () const
{
  std::stringstream ss;
  ss << std::setw(17) << std::fixed << std::setprecision(2) << tracked_oob_risk.back();
  
  return ss.str();
}





// TimeLogger:
// -----------------------

/**
 * \brief Default constructor of class `TimeLogger`
 * 
 * \param is_a_stopper0 `bool` which specifies if the logger is used as stopper
 * \param max_time `unsigned int` maximal time for training (just used if logger 
 *   is a stopper)
 * \param time_unit `std::string` of the unit used for measuring, allowed are 
 *   `minutes`, `seconds` and `microseconds`
 */

TimeLogger::TimeLogger (const bool& is_a_stopper0, const unsigned int& max_time, 
  const std::string& time_unit)
  : max_time ( max_time ),
    time_unit ( time_unit )
{
  is_a_stopper = is_a_stopper0;
}

/**
 * \brief Log current step of compboost iteration for class `TimeLogger`
 * 
 * This functions loggs dependent on `time_unit` the elapsed time at the
 * current iteration.
 * 
 * \param current_iteration `unsigned int` of current iteration 
 * \param response `arma::vec` of the given response used for training
 * \param prediction `arma::vec` actual prediction of the boosting model at 
 *   iteration `current_iteration`
 * \param used_blearner `Baselearner*` pointer to the selected baselearner in 
 *   iteration `current_iteration`
 * \param offset `double` of the overall offset of the training
 * \param learning_rate `double` lerning rate of the `current_iteration`
 * 
 */

void TimeLogger::logStep (const unsigned int& current_iteration, const arma::vec& response, 
  const arma::vec& prediction, blearner::Baselearner* used_blearner, const double& offset, 
  const double& learning_rate)
{
  if (current_time.size() == 0) {
    init_time = std::chrono::steady_clock::now();
  }
  if (time_unit == "minutes") {
    current_time.push_back(std::chrono::duration_cast<std::chrono::minutes>(std::chrono::steady_clock::now() - init_time).count());
  } 
  if (time_unit == "seconds") {
    current_time.push_back(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - init_time).count());
  } 
  if (time_unit == "microseconds") {
    current_time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - init_time).count());
  }
}

/**
 * \brief Stop criteria is fulfilled if the passed time exceeds `max_time`
 * 
 * The stop criteria here is quite simple. For the current iteration \f$i\f$ it 
 * is triggered if 
 * \f[
 *   \mathrm{current_time}_i > \mathrm{max_time}
 * \f]
 * 
 * \returns `bool` which tells if the stopping criteria is reached or not 
 *   (if the logger isn't a stopper then this is always false)
 */

bool TimeLogger::reachedStopCriteria () const
{
  bool stop_criteria_is_reached = false;
  
  if (is_a_stopper) {
    if (current_time.back() >= max_time) {
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

arma::vec TimeLogger::getLoggedData () const
{
  // Cast integer vector to double:
  std::vector<double> seconds_double (current_time.begin(), current_time.end());
  
  arma::vec out (seconds_double);
  return out;
}

/**
 * \brief Clear the logger data
 * 
 * This is an important thing which is called every time in front of retraining 
 * the model. If we don't clear the data, the new iterations are just pasted at
 * the end of the existing vectors which couses some troubles. 
 */

void TimeLogger::clearLoggerData ()
{
  current_time.clear();
}

/**
 * \brief Print the head of the trace which is printed to the console
 * 
 * \returns `std::string` which is used to initialize the header of the trace
 */

std::string TimeLogger::initializeLoggerPrinter () const
{
  std::stringstream ss;
  ss << std::setw(17) << time_unit;
  
  return ss.str();
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

std::string TimeLogger::printLoggerStatus () const
{
  std::stringstream ss;
  ss << std::setw(17) << std::fixed << std::setprecision(2) << current_time.back();
  
  return ss.str();
}

} // namespace logger
