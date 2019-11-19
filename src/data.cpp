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

void Data::setDataType (const std::string& data_type0)
{
  data_type = data_type0;
}


// -------------------------------------------------------------------------- //
// Data implementations:
// -------------------------------------------------------------------------- //

// InMemoryData:
// -----------------------

InMemoryData::InMemoryData () {}

InMemoryData::InMemoryData (const arma::mat& raw_data, const std::string& data_identifier0)
{
  data_mat = raw_data;
  data_identifier = data_identifier0;
}

void InMemoryData::setData (const arma::mat& transformed_data) { data_mat = transformed_data; }
arma::mat InMemoryData::getData () const { return data_mat; }

InMemoryData::~InMemoryData () {}

} // namespace data
