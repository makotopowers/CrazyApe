//============================================================================
// RSS implementation using Eigen library
// Makoto Powers
//
// Residual sum of squares is the sum of the squared error between the predicted
// values and the true values. The RSS can be computed using the following formula:
//
// RSS = sum_i (y_i - y_hat_i)^2
//
// With y_i the true values and y_hat_i the predicted values.
//============================================================================

#pragma once

//============================================================================
// INCLUDES
#include <Eigen/Dense>
#include <vector>

//============================================================================

namespace Metrics {

class RSS {

  //   @brief RSS class
  //
  //   This class implements residual sum of squares using Eigen library. The RSS
  //   can be computed using the compute method.

 public:
  RSS();
  ~RSS();

  double compute(const Eigen::VectorXd& y, const Eigen::VectorXd& y_hat);
  std::vector<double> compute(const Eigen::VectorXd& y, const Eigen::MatrixXd& y_hat);
};

}  // namespace Metrics