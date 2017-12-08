// =========================================================================== #
//                                 ___.                          __            #
//        ____  ____   _____ ______\_ |__   ____   ____  _______/  |_          #
//      _/ ___\/  _ \ /     \\____ \| __ \ /  _ \ /  _ \/  ___/\   __\         #
//      \  \__(  <_> )  Y Y  \  |_> > \_\ (  <_> |  <_> )___ \  |  |           #
//       \___  >____/|__|_|  /   __/|___  /\____/ \____/____  > |__|           #
//           \/            \/|__|       \/                  \/                 #
//                                                                             #
// =========================================================================== #
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
//   The "CompboostWrapper" class which is exported to R by using the Rcpp
//   modules. For a tutorial see
//   <http://dirk.eddelbuettel.com/code/rcpp/Rcpp-modules.pdf>.
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
// =========================================================================== #

#include "compboost.h"

class CompboostWrapper
{
  public:

    // Constructors
    CompboostWrapper () {
      cboost::Compboost *obj = new cboost::Compboost();
    };

    // Member functions
    arma::vec GetResponse () {
      return obj.GetResponse();
    };
    void SetResponse (arma::vec response) {
      obj.SetResponse(response);
    };

  private:

    cboost::Compboost obj;
};

// --------------------------------------------------------------------------- #
// Rcpp module:
// --------------------------------------------------------------------------- #

RCPP_MODULE(compboost_module) {

  using namespace Rcpp;

  class_<CompboostWrapper> ("CompboostWrapper")

  .constructor ("Initialize CompboostWrapper (Compboost) object")

  .method ("GetResponse", &CompboostWrapper::GetResponse, "Get the response of the Compboost object")
  .method ("SetResponse", &CompboostWrapper::SetResponse, "Set the response of the Compboost object")
  ;
}
