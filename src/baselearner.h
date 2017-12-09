// // ========================================================================== \\
// //                                 ___.                          __           \\
// //        ____  ____   _____ ______\_ |__   ____   ____  _______/  |_         \\
// //      _/ ___\/  _ \ /     \\____ \| __ \ /  _ \ /  _ \/  ___/\   __\        \\
// //      \  \__(  <_> )  Y Y  \  |_> > \_\ (  <_> |  <_> )___ \  |  |          \\
// //       \___  >____/|__|_|  /   __/|___  /\____/ \____/____  > |__|          \\
// //           \/            \/|__|       \/                  \/                \\
// //                                                                            \\
// // ========================================================================== \\
// //
// // Compboost is free software: you can redistribute it and/or modify
// // it under the terms of the GNU General Public License as published by
// // the Free Software Foundation, either version 3 of the License, or
// // (at your option) any later version.
// // Compboost is distributed in the hope that it will be useful,
// // but WITHOUT ANY WARRANTY; without even the implied warranty of
// // MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// // GNU General Public License for more details.
// // You should have received a copy of the GNU General Public License
// // along with Compboost. If not, see <http://www.gnu.org/licenses/>.
// //
// // This file contains:
// // -------------------
// //
// //   "Baselearner" class
// //
// // Written by:
// // -----------
// //
// //   Daniel Schalk
// //   Institut für Statistik
// //   Ludwig-Maximilians-Universität München
// //   Ludwigstraße 33
// //   D-80539 München
// //
// //   https://www.compstat.statistik.uni-muenchen.de
// //
// // =========================================================================== #
//
// #ifndef BASELEARNER_H_
// #define BASELEARNER_H_
//
// #include <RcppArmadillo.h>
// #include <string>
//
// namespace blearner {
//
// class Baselearner
// {
// private:
//   arma::vec *response_ptr; // Pointer to the response
//   arma::mat data; // Data for the baselearner e.g. one variable for linear bl
//   std::string var_name; // Variable name, needed to know which variable are wrapped
//   std::string blearner_name; // Name of the baselearner
//   arma::vec parameter; // Vector of parameters. At the moment just numeric params
//
// public:
//   Baselearner(arma::vec, arma::mat, std::string, std::string, arma::vec);
// };
//
// } // namespace cboost
//
// #endif // BASELEARNER_H_