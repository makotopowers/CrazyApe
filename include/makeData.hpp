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
#include <vector>

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

  void set_true_function(const std::function<double(const Eigen::VectorXd&)>& true_function_p, int n_informative);
  void make_data_multivariate_normal(const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov, int samples);
  void make_data_uniform(const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov, int samples);
  DataSet get_data_set(int n_samples, int n_redundant, double noise_level);
  Eigen::MatrixXd get_underlying_data() const;

  void set_underlying_data(const Eigen::MatrixXd& data);
  std::vector<DataSet> train_test_split(const DataSet& data_set, double test_size = 0.2);

 private:
  std::default_random_engine generator;
  std::normal_distribution<double> normal_dist;
  std::uniform_real_distribution<double> uniform_dist;
  std::function<double(const Eigen::VectorXd&)> true_function;
  int n_informative;
  bool function_set;
  bool underlying;
  Eigen::MatrixXd underlying_data;
};

}  // namespace Simulation
}  // namespace Data
