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
//   The "LossWrapper" class which is exported to R by using the Rcpp
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

#include "loss_factory.h"

class LossWrapper
{
  private:
    // This one leads to a first definition of the class Loss with the empty
    // constructor.
    lossfactory::LossFactory obj;

  public:
    // Constructors
    LossWrapper ()
    {
      // Predefine the firstly created object with the actual wanted structure!
      lossfactory::LossFactory *obj_ptr = &obj;

      // If we dont use pointers here we create two different objects and always
      // use the first one, which was created by an empty constructor, which
      // doesn't contain the object we really want.
      *obj_ptr = lossfactory::LossFactory();
      
      // for debugging:
      std::cout << "Create new LossWrapper: " << &obj << std::endl;
    }
    LossWrapper (std::string loss_name)
    {
      // Predefine the firstly created object with the actual wanted structure!
      lossfactory::LossFactory *obj_ptr = &obj;
      
      // If we dont use pointers here we create two different objects and always
      // use the first one, which was created by an empty constructor, which
      // doesn't contain the object we really want.
      *obj_ptr = lossfactory::LossFactory(loss_name);
      
      // for debugging:
      std::cout << "Create new LossWrapper with std::string "
                << &obj
                << std::endl;
    }
    LossWrapper (std::string loss_type, Rcpp::Function lossFun, Rcpp::Function gradientFun, Rcpp::Function initFun)
    {
      lossfactory::LossFactory *obj_ptr = &obj;
      *obj_ptr = lossfactory::LossFactory(loss_type, lossFun, gradientFun, initFun);
    }
    
    // Member functions
    arma::vec CalcLoss (arma::vec &true_value, arma::vec &response)
    {
      return obj.CalcLoss(true_value, response);
    }
    
    arma::vec CalcGradient (arma::vec &true_value, arma::vec &response)
    {
      return obj.CalcGradient(true_value, response);
    }
    
    double ConstantInitializer (arma::vec &true_value)
    {
      return obj.ConstantInitializer(true_value);
    }
    
    std::string GetLossName ()
    {
      return obj.GetLossType();
    }
};

// --------------------------------------------------------------------------- #
// Rcpp module:
// --------------------------------------------------------------------------- #

RCPP_MODULE(loss_module) {
  
  using namespace Rcpp;
  
  class_<LossWrapper> ("LossWrapper")
    
    .constructor ("Initialize loss as quadratic")
    .constructor <std::string> ("Initialize LossWrapper (Loss) object")
  .constructor <std::string, Rcpp::Function, Rcpp::Function, Rcpp::Function> ("Initialize own loss")

  .method ("CalcLoss", &LossWrapper::CalcLoss, "Get the loss")
  .method ("CalcGradient", &LossWrapper::CalcGradient, "Get the loss")
  .method ("ConstantInitializer", &LossWrapper::ConstantInitializer, "Initialize loss optimal")
  .method ("GetLossName", &LossWrapper::GetLossName, "Get the specific loss name")
  ;
}
