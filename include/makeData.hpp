//============================================================================
// Dataset creator using Eigen library
// Makoto Powers
//
// This class generates synthetic datasets for regression problems.
//============================================================================

#pragma once

//============================================================================
// INCLUDES
#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <random>

//============================================================================

namespace Data {
namespace Simulation {

typedef struct {
  Eigen::MatrixXd X;
  Eigen::VectorXd y;
} DataSet;

class MakeData {
  //   @brief MakeData class
  //
  //   This class generates synthetic datasets for regression problems. The class
  //   can generate datasets with a specified number of samples, features, and
  //   noise level. The class can also generate datasets with a specified number
  //   of informative features and redundant features.

 public:
  MakeData();
  ~MakeData();

  void set_true_function(const std::function<double(const Eigen::VectorXd&)>& true_function_p, const int n_informative);
  DataSet get_data_set(const int n_samples, const int n_redundant, const double noise_level);
  void CreateUnderlyingMultivariateNormal(const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov, const int samples);
  Eigen::MatrixXd get_multivariate_normal();

 private:
  std::default_random_engine generator;
  std::normal_distribution<double> normal_dist;
  std::function<double(const Eigen::VectorXd&)> true_function;
  int n_informative;
  std::shared_ptr<Eigen::MatrixXd> multivariate_normal;
};

}  // namespace Simulation
}  // namespace Data
