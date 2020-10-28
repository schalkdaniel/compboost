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
// ========================================================================== //

#include "loggerlist.h"

namespace loggerlist
{

LoggerList::LoggerList () {}

void LoggerList::registerLogger (std::shared_ptr<logger::Logger> logger)
{
  _logger_list.insert(logger_pair(logger->getLoggerId(), logger));
  if (logger->isStopper()) { _sum_of_stopper += 1; }
}

void LoggerList::printRegisteredLogger () const
{
  Rcpp::Rcout << "Registered Logger:\n";
  for (auto& it_logger : _logger_list) {
    Rcpp::Rcout << "\t>>" << it_logger.first << "<< Logger" << std::endl;
  }
}

logger_map LoggerList::getLoggerMap () const { return _logger_list; }

void LoggerList::clearMap () { _logger_list.clear(); }

bool LoggerList::getStopperStatus (const bool use_global_stop) const
{
  // Define variables to get the status of the algorithm:

  // Should the algorithm be stopped?
  bool stop_algorithm = false;
  // Get status for every registered logger:
  std::vector<bool> status;

  for (auto& it_logger : _logger_list) {
    status.push_back(it_logger.second->reachedStopCriteria());
  }
  unsigned int status_sum = std::accumulate(status.begin(), status.end(), 0);

  // Check if global stop (all stopper has to be true) or local stop (it is
  // sufficient to have just one stopper saying true):
  if (use_global_stop) {
    if (status_sum == _sum_of_stopper) { stop_algorithm = true; }
  } else {
    if (status_sum >= 1) { stop_algorithm = true; }
  }
  return !stop_algorithm;
}

logger_data LoggerList::getLoggerData () const
{
  arma::mat out_matrix;
  std::vector<std::string> logger_names;

  for (auto& it_logger : _logger_list) {
    out_matrix = arma::join_rows(out_matrix, it_logger.second->getLoggedData());
    logger_names.push_back(it_logger.first);
  }
  return logger_data(logger_names, out_matrix);
}

void LoggerList::logCurrent (const unsigned int current_iteration, const std::shared_ptr<response::Response>& sh_ptr_response,
  const std::shared_ptr<blearner::Baselearner>& sh_ptr_blearner, const double learning_rate, const double step_size,
  const std::shared_ptr<optimizer::Optimizer>& sh_ptr_optimizer)
{
  for (auto& it_logger : _logger_list) {
    it_logger.second->logStep(current_iteration, sh_ptr_response, sh_ptr_blearner,
      learning_rate, step_size, sh_ptr_optimizer);
  }
}

void LoggerList::printLoggerStatus (const double current_risk) const
{
  std::stringstream printer;
  bool print_risk = true;

  for (auto& it_logger : _logger_list) {
    printer << it_logger.second->printLoggerStatus() << "   ";
    if (print_risk) {
      printer << "risk = " << std::setprecision(2) << current_risk << "  ";
      print_risk = false;
    }
  }
  Rcpp::Rcout << printer.str() << std::endl;
}

void LoggerList::prepareForRetraining (const unsigned int new_max_iters)
{
  bool has_iteration_logger = false;

  for (auto& it_logger : _logger_list) {
    it_logger.second->setIsStopper(false);

    if (it_logger.second->getLoggerType() == "iteration") {
      std::static_pointer_cast<logger::LoggerIteration>(it_logger.second)->updateMaxIterations(new_max_iters);
      it_logger.second->setIsStopper(true);
      has_iteration_logger = true;
    }
    if (it_logger.second->getLoggerType() == "time") {
      std::static_pointer_cast<logger::LoggerTime>(it_logger.second)->reInitializeTime();
    }
  }
  if (! has_iteration_logger) {
    std::shared_ptr<logger::Logger> new_logger = std::make_shared<logger::LoggerIteration>("iters_re", true, new_max_iters);
    _logger_list.insert(logger_pair("_iteration", new_logger));
  }
}

void LoggerList::clearLoggerData ()
{
  for (auto& it_logger : _logger_list) {
    it_logger.second->clearLoggerData();
  }
}

LoggerList::~LoggerList () {}

} // namespace loggerlist
