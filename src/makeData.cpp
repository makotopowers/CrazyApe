//============================================================================
// Implementation of the makeData class.
// Makoto Powers
//============================================================================

//============================================================================
// INCLUDES
#include "makeData.hpp"

//============================================================================

namespace Data {
namespace Simulation {

MakeData::MakeData() : generator(std::random_device()()), normal_dist(0.0, 1.0), underlying(0) {}

MakeData::~MakeData() {}

void MakeData::set_true_function(const std::function<double(const Eigen::VectorXd&)>& true_function_p, int n_informative) {
  // @brief Set the true function to generate the target values
  //
  // @param true_function_p: true function to generate the target values
  // @param n_informative: number of informative features

  this->true_function = true_function_p;
  this->n_informative = n_informative;

  return;
}

DataSet MakeData::get_data_set(int n_samples, int n_redundant, double noise_level) {
  // @brief Generate a synthetic dataset
  //
  // @param n_samples: number of samples
  // @param n_redundant: number of redundant features
  // @param noise_level: standard deviation of the noise
  //
  // @return data_set: synthetic dataset

  Eigen::MatrixXd informative_features;

  if (underlying == 0) {
    informative_features = Eigen::MatrixXd::Random(n_samples, n_informative);
  } else {
    assert(n_informative == underlying_data.cols());
    informative_features = underlying_data;
    n_samples = informative_features.rows();
  }

  int n_features = n_informative + n_redundant;

  // calculate y using the true function
  Eigen::VectorXd y = Eigen::VectorXd::Zero(n_samples);
  for (int i = 0; i < n_samples; i++) {
    y(i) = true_function(informative_features.row(i).transpose());
  }

  // create a matrix of random features of size n_redundant x n_samples
  Eigen::MatrixXd redundant_features = Eigen::MatrixXd::Random(n_samples, n_redundant);

  // concatenate the informative and redundant features
  Eigen::MatrixXd X = Eigen::MatrixXd(n_samples, n_features);
  X.block(0, 0, n_samples, n_informative) = informative_features;
  X.block(0, n_informative, n_samples, n_redundant) = redundant_features;

  // add noise to the target values
  for (int i = 0; i < n_samples; i++) {
    y(i) += noise_level * normal_dist(generator);
  }

  Data::Simulation::DataSet data_set;
  data_set.X = X;
  data_set.y = y;

  return data_set;
}

void MakeData::make_data_multivariate_normal(const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov, int samples) {
  // @brief Set the underlying multivariate normal distribution
  //
  // @param mean: mean of the multivariate normal distribution
  // @param cov: covariance matrix of the multivariate normal distribution
  // @param samples: number of samples

  assert(cov.rows() == mean.size());
  assert(cov.cols() == mean.size());

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(mean.size(), samples);
  for (int i = 0; i < mean.size(); i++) {
    for (int j = 0; j < samples; j++) {
      A(i, j) = normal_dist(generator);
    }
  }

  Eigen::MatrixXd L = cov.llt().matrixL();
  Eigen::MatrixXd B = (L * A).transpose();

  assert(B.rows() == samples);
  assert(B.cols() == mean.size());

  this->underlying_data = B;
  this->underlying = 1;

  return;
}

Eigen::MatrixXd MakeData::get_underlying_data() const {
  // @brief Get the underlying data
  //
  // @return underlying_data: underlying data

  return this->underlying_data;
}

void MakeData::set_underlying_data(const Eigen::MatrixXd& data) {
  // @brief Set the underlying data
  //
  // @param data: underlying data

  this->underlying_data = data;
  this->underlying = 1;

  return;
}

}  // namespace Simulation
}  // namespace Data