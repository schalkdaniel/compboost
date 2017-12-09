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
// 
// #include "baselearner.h"
// #include <iostream>
// 
// namespace blearner {
// 
// // --------------------------------------------------------------------------- #
// // Constructors:
// // --------------------------------------------------------------------------- #
// 
// Baselearner::Baselearner (arma::vec response_ptr0, arma::mat data0,
//   std::string var_name0, std::string blearner_name0, arma::vec parameter0)
// {
//   std::cout << "A new Baselearner object has ben created!" << std::endl;
// 
//   response_ptr = &response_ptr0;
//   data = data0;
//   var_name = var_name0;
//   blearner_name = blearner_name0;
//   parameter = parameter0;
// }
// 
// // --------------------------------------------------------------------------- #
// // Member functions:
// // --------------------------------------------------------------------------- #
// 
// void Compboost::SetResponse (arma::vec response0)
// {
//   response = response0;
// }
// 
// arma::vec Compboost::GetResponse ()
// {
//   return response;
// }
// 
// } // namespace cboost