#include <Rcpp.h>

#include "compboost.h"

class CompboostWrapper {
  public:
    // Constructors
    CompboostWrapper () {
      cboost::Compboost *obj = new cboost::Compboost();
    };
    CompboostWrapper (std::string name) {
      cboost::Compboost *obj = new cboost::Compboost(name);
    };

    // Member functions
    std::string GetName () {
      return obj.GetName();
    };
    void SetName (std::string name) {
      obj.SetName(name);
    };

  private:
    cboost::Compboost obj;
};


RCPP_MODULE(compboost_module) {

  using namespace Rcpp;

  class_<CompboostWrapper> ("CompboostWrapper")

  .constructor ("Initialize CompboostWrapper (Compboost) object")
  .constructor <std::string> ("Initialize CompboostWrapper (Compboost) with name")

  .method ("GetName", &CompboostWrapper::GetName, "Get the name of the Compboost object")
  .method ("SetName", &CompboostWrapper::SetName, "Set the name of the Compboost object")
  ;
}
