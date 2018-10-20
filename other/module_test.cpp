#include <Rcpp.h>
#include <sstream>
#include <string>

// [[Rcpp::export]]
void summaryArgumentList (Rcpp::List list) 
{
	std::vector<std::string> list_names = list.names();
	for (unsigned int i = 0; i < list.size(); i++) {
		Rcpp::Rcout << list_names[i] << "\t:\t" << Rcpp::toString(list[list_names[i]]) << std::endl;
	}
}

void thisIsSomething (int param_a, std::string param_b)
{
	std::cout << "Int: " << param_a << "; String: " << param_b << std::endl;
}

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

Rcpp::List argHandler (Rcpp::List internal_list, Rcpp::List matching_list)
{
	std::vector<std::string> internal_list_names = internal_list.names();
	std::vector<std::string> matching_list_names = matching_list.names();
	std::vector<std::string> unused_args;

	int arg_type_internal;
	int arg_type_match;

	for (unsigned int i = 0; i < matching_list_names.size(); i++) {
		Rcpp::Rcout << matching_list_names[i] << " in list: " << stringInNames(matching_list_names[i], internal_list_names) << std::endl;
		if (stringInNames(matching_list_names[i], internal_list_names)) {
			// Check element type:
			arg_type_internal = TYPEOF(internal_list[matching_list_names[i]]);
			arg_type_match    = TYPEOF(matching_list[matching_list_names[i]]);

			// Int as 13 and double as 14 are both ok, so set both to 13 if they are 14:
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

class Test {

private:
	Rcpp::List internal_arg_list = Rcpp::List::create(Rcpp::Named("param_a") = 2, Rcpp::Named("param_b") = "bla");

public:
	Test () {};
	Test (Rcpp::List arg_list) {
		internal_arg_list = argHandler(internal_arg_list, arg_list);
	};
	void doSomethingWithList () {
		thisIsSomething(internal_arg_list["param_a"], internal_arg_list["param_b"]);
	};
	Rcpp::List getList () {
		return internal_arg_list;
	};
	// bool checkParameterTypes (arg_list) {

	// 	int test_int = arg_list["param_a"];
	// 	std::string test_str = arg_list["param_b"];

	// 	return true;
	// }
	// Rcpp::List mylist = Rcpp::List::create(Rcpp::Named("degree") = 1,
 //                          Rcpp::Named("intercept") = TRUE);
};

RCPP_MODULE(test_module) {
	using namespace Rcpp;
	class_<Test>("Test")
	.constructor()
	.constructor<Rcpp::List>()
	.method("doSomethingWithList", &Test::doSomethingWithList)
	.method("getList", &Test::getList)
	;
}