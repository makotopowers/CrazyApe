//============================================================================
// Logistic Regression implementation using Eigen library
// Makoto Powers
//
// Logistic regression is a classification model that models the probability of a
// binary outcome. The model can be posed as the following problem:
//
// beta* = argmin -sum_i (y_i * log(p_i) + (1 - y_i) * log(1 - p_i)) + lambda * ||beta||^2
//
// With X an n x p matrix, y an n x 1 vector, beta a p x 1 vector, and lambda a
// scalar. lambda can be seen as a regularization parameter. When lambda is 0, the
// problem is equivalent to logistic regression.
//
// In Bayesian terms, the logistic regression problem is equivalent to finding the
// mode of the posterior distribution of beta, with a Gaussian prior on beta with
// mean 0 and covariance matrix (1/lambda) * I.
//============================================================================

#pragma once

//============================================================================
// INCLUDES
#include <Eigen/Dense>
#include <vector>

//============================================================================

namespace Modelling {
namespace Classification {

class LogisticRegression {

  //   @brief Logistic Regression class
  //
  //   This class implements logistic regression using Eigen library. Logistic regression
  //   is a classification model that models the probability of a binary outcome. The model
  //   can be fit using the fit method, and the coefficients can be accessed using the get_beta
  //   method.

 public:
  LogisticRegression();
  ~LogisticRegression();

  Eigen::VectorXd fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const double lambda);

  Eigen::VectorXd get_beta();

 private:
  Eigen::VectorXd beta;
};

}  // namespace Classification
}  // namespace Modelling