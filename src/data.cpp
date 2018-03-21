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
//   
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

#include "data.h"

namespace data
{

Data::Data () {}

void Data::setDataIdentifier (const std::string& new_data_identifier)
{
  data_identifier = new_data_identifier;
}

std::string Data::getDataIdentifier () const 
{
  return data_identifier;
}



// -------------------------------------------------------------------------- //
// Data implementations:
// -------------------------------------------------------------------------- //

// InMemoryData:
// -----------------------

InMemoryData::InMemoryData () {}

// InMemoryData::InMemoryData (const arma::vec& raw_data, const std::string& data_identifier0)
// {
//   Rcpp::Rcout << "Vector Initializer" << std::endl;
//   data_mat_ptr    = &raw_data;
//   data_identifier = data_identifier0;
// }

InMemoryData::InMemoryData (const arma::mat& raw_data, const std::string& data_identifier0)
{
  data_mat_ptr    = &raw_data;
  data_identifier = data_identifier0;
}

void InMemoryData::setData (const arma::mat& transformed_data) 
{
  data_mat = transformed_data;
}

arma::mat InMemoryData::getData () const
{
  // Give data depending on source (by reference) or target (by value):
  if (data_mat_ptr == NULL) {
    return data_mat;
  } else {
    return *data_mat_ptr;
  }
}

InMemoryData::~InMemoryData () {
  // Rcpp::Rcout << "Delete Data" << std::endl;
}

} // namespace data