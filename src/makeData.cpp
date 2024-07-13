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

MakeData::MakeData() : generator(std::random_device()()), normal_dist(0.0, 1.0) {}

MakeData::~MakeData() {}

void MakeData::set_true_function(const std::function<double(const Eigen::VectorXd&)>& true_function_p, const int n_informative) {
  // @brief Set the true function to generate the target values
  //
  // @param true_function_p: true function to generate the target values

  this->true_function = true_function_p;
  this->n_informative = n_informative;

  return;
}

DataSet MakeData::get_data_set(const int n_samples, const int n_redundant, const double noise_level) {
  // @brief Generate a synthetic dataset
  //
  // @param n_samples: number of samples
  // @param true_function_p: true function to generate the target values
  // @param n_informative: number of informative features
  // @param n_redundant: number of redundant features
  // @param noise_level: standard deviation of the noise
  //
  // @return X: n x p matrix of features
  // @return y: n x 1 vector of target values

  int n_features = n_informative + n_redundant;

  // create a matrix of random features of size n_informative x n_samples
  Eigen::MatrixXd informative_features = Eigen::MatrixXd::Random(n_samples, n_informative);

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

}  // namespace Simulation
}  // namespace Data