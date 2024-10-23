//============================================================================
// K Nearest Neigbors implementation using Eigen library
// Makoto Powers
//
// K Nearest Neighbors is a non-parametric method for classification and regression.
// The method is based on the idea that similar data points should have similar
// target values. The method works by finding the k nearest neighbors to a given
// data point, and then predicting the target value based on the target values of
// the k nearest neighbors.
//
// This implementation is done using Kd-trees.
//============================================================================

#pragma once

//============================================================================
// INCLUDES
#include <Eigen/Dense>

//============================================================================

namespace Modelling {
namespace NonParametric {

class KNN {

  //   @brief K Nearest Neighbors class
  //
  //   This class implements K Nearest Neighbors using Eigen library. K Nearest
  //   Neighbors is a non-parametric method for classification and regression. The
  //   model can be fit using the fit method, and the target values can be predicted
  //   using the predict method.

 public:
  KNN();
  ~KNN();

  void fit(const Eigen::MatrixXd& X_p, const Eigen::VectorXd& y_p, int k_p);
  Eigen::VectorXd predict(const Eigen::MatrixXd& X_p) const;

 private:
  Eigen::MatrixXd X;
  Eigen::VectorXd y;
  int k;
};

}  // namespace NonParametric
}  // namespace Modelling