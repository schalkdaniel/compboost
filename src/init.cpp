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
// it under the terms of the LGPL-3 License.
// Compboost is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// LGPL-3 License for more details. You should have received a copy of
// the license along with compboost.
//
// =========================================================================== #

#include "init.h"

namespace init {

typedef std::shared_ptr<data::Data> sdata;
typedef std::shared_ptr<data::BinnedData> sbindata;

PolynomialAttributes::PolynomialAttributes ()
{ }

PolynomialAttributes::PolynomialAttributes (const unsigned int _degree, const bool _use_intercept)
  : degree        ( _degree ),
    use_intercept ( _use_intercept )
{ }

PolynomialAttributes::PolynomialAttributes (const json& j)
  : df            ( j["df"].get<double>() ),
    penalty       ( j["penalty"].get<double>() ),
    penalty_mat   ( saver::jsonToArmaMat(j["penalty_mat"]) ),
    degree        ( j["degree"].get<unsigned int>() ),
    use_intercept ( j["use_intercept"].get<bool>() ),
    bin_root      ( j["bin_root"].get<unsigned int>() )
{ }

json PolynomialAttributes::toJson () const
{
  json j = {
    {"Class", "PolynomialAttributes"},

    {"df",            df},
    {"penalty",       penalty},
    {"penalty_mat",   saver::armaMatToJson(penalty_mat)},
    {"degree",        degree},
    {"use_intercept", use_intercept},
    {"bin_root",      bin_root}
  };
  return j;
}

PSplineAttributes::PSplineAttributes ()
{ }

PSplineAttributes::PSplineAttributes (const json& j)
  : df                  ( j["df"].get<double>() ),
    penalty             ( j["penalty"].get<double>() ),
    penalty_mat         ( saver::jsonToArmaMat(j["penalty_mat"]) ),
    degree              ( j["degree"].get<unsigned int>() ),
    n_knots             ( j["n_knots"].get<unsigned int>() ),
    differences         ( j["differences"].get<unsigned int>() ),
    use_sparse_matrices ( j["use_sparse_matrices"].get<bool>() ),
    bin_root            ( j["bin_root"].get<unsigned int>() ),
    knots               ( saver::jsonToArmaMat(j["knots"]) )
{ }

json PSplineAttributes::toJson () const
{
  json j = {
    {"Class", "PSplineAttributes"},

    {"df",                  df},
    {"penalty",             penalty},
    {"penalty_mat",         saver::armaMatToJson(penalty_mat)},
    {"degree",              degree},
    {"n_knots",             n_knots},
    {"differences",         differences},
    {"use_sparse_matrices", use_sparse_matrices},
    {"bin_root",            bin_root},
    {"knots",               saver::armaMatToJson(knots)}
  };
  return j;
}

RidgeAttributes::RidgeAttributes ()
{ }

RidgeAttributes::RidgeAttributes (const json& j)
  : df          ( j["df"].get<double>() ),
    penalty     ( j["penalty"].get<double>() ),
    penalty_mat ( saver::jsonToArmaMat(j["penalty_mat"]) ),
    dictionary  ( j["dictionary"].get<std::map<std::string, unsigned int>>() )
{ }

json RidgeAttributes::toJson () const
{
  json j = {
    {"Class", "RidgeAttributes"},

    {"df",          df},
    {"penalty",     penalty},
    {"penalty_mat", saver::armaMatToJson(penalty_mat)},
    {"dictionary",  dictionary}
  };
  return j;
}

BinaryAttributes::BinaryAttributes ()
{ }

BinaryAttributes::BinaryAttributes (const json& j)
  : cls ( j["cls"] )
{ }

json BinaryAttributes::toJson () const
{
  json j = {
    {"Class", "BinaryAttributes"},
    {"cls",   cls}
  };
  return j;
}

TensorAttributes::TensorAttributes ()
{ }

TensorAttributes::TensorAttributes (const json& j)
  : penalty ( j["penalty"].get<double>() )
{ }

json TensorAttributes::toJson () const
{
  json j = {
    {"Class",   "TensorAttributes"},
    {"penalty", penalty}
  };
  return j;
}

CenteredAttributes::CenteredAttributes ()
{ }

CenteredAttributes::CenteredAttributes (const json& j)
  : rotation ( saver::jsonToArmaMat(j["rotation"]) )
{ }

json CenteredAttributes::toJson () const
{
  json j = {
    {"Class",    "CenteredAttributes"},
    {"rotation", saver::armaMatToJson(rotation)}
  };
  return j;
}

json CustomCppAttributes::toJson () const
{
  json j = { {"Class","CenteredAttributes"} };
  throw std::logic_error("Cannot save CustomCppAttributes to JSON.");
  return j;
}


sbindata initPolynomialData (const sdata& raw_data, const std::shared_ptr<PolynomialAttributes>& attributes)
{
  arma::mat mraw = raw_data->getData();
  // This is necessary to prevent the program from segfolds... whyever???
  // Copied from: http://lists.r-forge.r-project.org/pipermail/rcpp-devel/2012-November/004796.html
  try {
    if (mraw.n_cols > 1) {
      Rcpp::stop("Given data should just have one column.");
    }
  } catch ( std::exception &ex ) {
    forward_exception_to_r( ex );
  } catch (...) {
    ::Rf_error( "c++ exception (unknown reason) in initialization of data for BaselearnerPolynomialFactory" );
  }

  sbindata sh_ptr_bindata;
  if (attributes->bin_root == 0) { // don't use binning
    sh_ptr_bindata = std::make_shared<data::BinnedData>(raw_data->getDataIdentifier());
  } else {             // use binning
    arma::colvec bins = binning::binVectorCustom(mraw, attributes->bin_root, "linear");
    sh_ptr_bindata = std::make_shared<data::BinnedData>(raw_data->getDataIdentifier(), attributes->bin_root, mraw, bins);
    mraw = bins;
  }

  arma::mat   out;
  if (attributes->use_intercept) {
    out = arma::mat(mraw.n_rows, 1, arma::fill::ones);
  }
  for (unsigned int i = 0; i < attributes->degree; i++) {
    out = arma::join_rows(out, arma::pow(mraw, i+1));
  }

  sh_ptr_bindata->setDenseData(out);
  sh_ptr_bindata->setMinMax(raw_data->getMinMax());
  return sh_ptr_bindata;
}

sbindata initPSplineData (const sdata& raw_data, const std::shared_ptr<PSplineAttributes>& attributes)
{
  auto mraw = raw_data->getData();
  // This is necessary to prevent the program from segfolds... whyever???
  // Copied from: http://lists.r-forge.r-project.org/pipermail/rcpp-devel/2012-November/004796.html
  try {
    if (mraw.n_cols > 1) {
      Rcpp::stop("Given data should just have one column.");
    }
  } catch ( std::exception &ex ) {
    forward_exception_to_r( ex );
  } catch (...) {
    ::Rf_error( "c++ exception (unknown reason) in initialization of data for  BaselearnerPSplineFactory" );
  }

  sbindata sh_ptr_bindata;
  if (attributes->bin_root == 0) { // don't use binning
    sh_ptr_bindata = std::make_shared<data::BinnedData>(raw_data->getDataIdentifier());
  } else {             // use binning
    arma::colvec bins = binning::binVectorCustom(mraw, attributes->bin_root, "linear");
    sh_ptr_bindata = std::make_shared<data::BinnedData>(raw_data->getDataIdentifier(), attributes->bin_root, mraw, bins);
    mraw = bins;
  }
  sh_ptr_bindata->setSparseData(splines::createSparseSplineBasis (mraw, attributes->degree, attributes->knots).t());
  sh_ptr_bindata->setMinMax(raw_data->getMinMax());

  return sh_ptr_bindata;
}

sdata initRidgeData (const sdata& raw_data, const std::shared_ptr<RidgeAttributes>& attributes)
{
  auto sh_ptr_cdata = std::static_pointer_cast<data::CategoricalDataRaw>(raw_data);
  auto chr_classes = sh_ptr_cdata->getRawData();
  arma::urowvec classes(chr_classes.size(), arma::fill::zeros);
  arma::urowvec row_idx(chr_classes.size(), arma::fill::zeros);
  unsigned int k = 0;
  for (unsigned int i = 0; i < chr_classes.size(); i++) {
    auto it = attributes->dictionary.find(chr_classes.at(i));
    if (it != attributes->dictionary.end()) {
      classes(i) = it->second;
      row_idx(k) = i;
      k += 1;
    }
  }
  classes = classes.head_cols(k);
  row_idx = row_idx.head_cols(k);
  arma::vec fill(k, arma::fill::ones);

  // Initialize sparse data matrix as (transposed) binary matrix (p x n).
  // Switching row_idx and col_idx gives the transposed p x n matrix:
  arma::umat locations = arma::join_cols(classes, row_idx);

  auto sh_ptr_data = std::make_shared<data::InMemoryData>(raw_data->getDataIdentifier());
  sh_ptr_data->setSparseData(arma::sp_mat(locations, fill, attributes->dictionary.size(), chr_classes.size()));
  sh_ptr_data->setMinMax(std::vector<double>{1, double(attributes->dictionary.size())});
  return sh_ptr_data;
}

sdata initBinaryData (const sdata& raw_data, const std::shared_ptr<BinaryAttributes>& attributes)
{
  auto sh_ptr_cdata = std::static_pointer_cast<data::CategoricalDataRaw>(raw_data);
  auto chr_classes = sh_ptr_cdata->getRawData();

  arma::urowvec row_idx(chr_classes.size(), arma::fill::zeros);
  unsigned int k = 0;
  for (unsigned int i = 0; i < chr_classes.size(); i++) {
    if (attributes->cls == chr_classes.at(i)) {
      row_idx(k) = i;
      k += 1;
    }
  }
  arma::urowvec classes(k, arma::fill::zeros);
  row_idx = row_idx.head_cols(k);
  arma::vec fill(k, arma::fill::ones);

  arma::umat locations = arma::join_cols(classes, row_idx);

  arma::sp_mat dm = arma::sp_mat(locations, fill, 1, chr_classes.size());

  auto sh_ptr_data = std::make_shared<data::InMemoryData>(raw_data->getDataIdentifier() + "_" + attributes->cls);
  sh_ptr_data->setSparseData(dm);
  sh_ptr_data->setMinMax(std::vector<double>{0, 1});
  return sh_ptr_data;
}


sdata initTensorData (const sdata& data1, const sdata& data2)
{
  // Extract mat from data1 and data2
  arma::mat tensor;
  arma::sp_mat tensor_sp;

  auto sh_ptr_data = std::make_shared<data::InMemoryData>(data1->getDataIdentifier() + "_" + data2->getDataIdentifier());
  if (data1->usesSparseMatrix() || data2->usesSparseMatrix()) {

      arma::sp_mat bl1_mat;
    if (data1->usesSparseMatrix()) {
      bl1_mat = data1->getSparseData().t();
    } else {
      bl1_mat = data1->getDenseData();
    }

    arma::sp_mat bl2_mat;
    if (data2->usesSparseMatrix()) {
      bl2_mat = data2->getSparseData().t();
    } else {
      bl2_mat = data2->getDenseData();
    }
    arma::sp_mat blc_mat = tensors::rowWiseKroneckerSparse(bl1_mat, bl2_mat);
    sh_ptr_data->setSparseData(blc_mat.t());
  } else {
    arma::mat bl1_mat = data1->getDenseData();
    arma::mat bl2_mat = data2->getDenseData();

    arma::mat blc_mat = tensors::rowWiseKronecker(bl1_mat, bl2_mat);
    sh_ptr_data->setDenseData(blc_mat);
  }
  std::vector<double> minmax1 = data1->getMinMax();
  std::vector<double> minmax2 = data2->getMinMax();
  minmax1.insert(std::end(minmax1), std::begin(minmax2), std::end(minmax2));

  sh_ptr_data->setMinMax(minmax1);
  return sh_ptr_data;
}

//sdata initCenteredData (const sbindata& bl_data, const std::shared_ptr<CenteredAttributes>& attributes)
sbindata initCenteredData (const sdata& bl_data, const std::shared_ptr<CenteredAttributes>& attributes)
{
  arma::mat temp;
  if (bl_data->usesSparseMatrix()) {
    temp = (attributes->rotation.t() * bl_data->getSparseData()).t();
  } else {
    temp = bl_data->getDenseData() * attributes->rotation;
  }
  sbindata sh_ptr_bindata = std::make_shared<data::BinnedData>(bl_data->getDataIdentifier() + "_" + bl_data->getDataIdentifier()); //, attributes->bin_root, mraw, bins);
  sh_ptr_bindata->setDenseData(temp);
  sh_ptr_bindata->setMinMax(bl_data->getMinMax());
  return sh_ptr_bindata;
}

sdata initCustomData (const sdata& raw_data, Rcpp::Function instantiateDataFun)
{
  auto sh_ptr_data = std::make_shared<data::InMemoryData>(raw_data->getDataIdentifier());
  arma::mat temp =  Rcpp::as<arma::mat>(instantiateDataFun(raw_data->getDenseData()));
  sh_ptr_data->setDenseData(temp);
  return sh_ptr_data;
}

typedef arma::mat (*instantiateDataFunPtr) (const arma::mat& X);
sdata initCustomCppData (const sdata& raw_data, const std::shared_ptr<CustomCppAttributes>& attributes)
{
  auto sh_ptr_data = std::make_shared<data::InMemoryData>(raw_data->getDataIdentifier());
  sh_ptr_data->setDenseData(attributes->instantiateDataFun(raw_data->getDenseData()));
  return sh_ptr_data;
}


} // namespace init


