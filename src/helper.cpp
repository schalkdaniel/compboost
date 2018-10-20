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
// Written by:
// -----------
//
//   Daniel Schalk
//   Department of Statistics
//   Ludwig-Maximilians-University Munich
//   Ludwigstrasse 33
//   D-80539 München
//
//   https://www.compstat.statistik.uni-muenchen.de
//
//   Contact
//   e: contact@danielschalk.com
//   w: danielschalk.com
//
// =========================================================================== #

#include <Rcpp.h>
#include <sstream>
#include <string>

/**
 * \brief Check if a string occurs within a string vector
 * 
 * This function just takes a string, iterates over a given vector of strings,
 * and returns true if the string occurs within the vector. This function is 
 * used to check the argument name match up.
 * 
 * \param str `str::string` String for the lookup.
 *   
 * \param differences `std::vector<std::string>` Vector of strings which we 
 '   want to check if str occurs..
 * 
 * \returns `bool` boolean if the string occurs within the vector. 
 */
bool stringInNames (std::string str, std::vector<std::string> names)
{
  bool string_in_names = false;
  for (unsigned int i = 0; i < names.size(); i++) {
    if (str == names[i]) {
      string_in_names = true;
      return string_in_names;
    }
  }
  return string_in_names;
}


/**
 * \brief Check and set list arguments
 * 
 * This function matches to lists to update an internal list with a new
 * match-up list. This function checks which elements are available, 
 * if they occurs in the internal list, and if the underlying data types
 * matches. If so, this function replaces
 * the values of the internal list with the new values. If the new list 
 * contains unused elements, this function also throws a warning and prints
 * the unused ones.
 * 
 * \param internal_list `Rcpp::List` Internal list with default values.
 * \param matching_list `Rcpp::List` New list to replace default values.
 * \returns `Rcpp::List` of updated default values. 
 */
Rcpp::List argHandler (Rcpp::List internal_list, Rcpp::List matching_list)
{
  std::vector<std::string> internal_list_names = internal_list.names();
  std::vector<std::string> matching_list_names = matching_list.names();
  std::vector<std::string> unused_args;

  int arg_type_internal;
  int arg_type_match;

  for (unsigned int i = 0; i < matching_list_names.size(); i++) {
    if (stringInNames(matching_list_names[i], internal_list_names)) {

      // Check element type:
      arg_type_internal = TYPEOF(internal_list[matching_list_names[i]]);
      arg_type_match    = TYPEOF(matching_list[matching_list_names[i]]);

      // Int as 13 and double as 14 are both ok, so set both to 13 (int) if they are 14 (double). The
      // as conversion of Rcpp handles to set them correctly later:
      if (arg_type_internal == 14) { arg_type_internal -= 1; }
      if (arg_type_match == 14) { arg_type_match -= 1; }

      if (arg_type_internal == arg_type_match) {
        internal_list[matching_list_names[i]] = matching_list[matching_list_names[i]];
      } else {
        Rcpp::stop("Argument types for \"" + matching_list_names[i] + "\" does not match. Maybe you should take a look at the documentation.");
      }
    } else {
      unused_args.push_back(matching_list_names[i]);
    }
  }
  if (unused_args.size() > 0) {
    std::stringstream message;
    message << "Unused arguments ";

    for (unsigned int i = 0; i < unused_args.size(); i++) {
      message << unused_args[i];
      if (i < unused_args.size() - 1) {
        message << ", ";
      }
    }
    message << " in list.";
    Rcpp::warning(message.str());
  }
  return internal_list;
}