#include "compboost.h"

#include <iostream>

namespace cboost {

Compboost::Compboost ()
{
  std::cout << "A new Compboost object has ben created!" << std::endl;
}

Compboost::Compboost (std::string name0)
{
  std::cout << "A new Compboost object with name "
            << name0
            << " has ben created!"
            << std::endl;

  name = name0;
}

void Compboost::SetName (std::string name0)
{
  name = name0;
}

std::string Compboost::GetName ()
{
  return name;
}

} // namespace cboost