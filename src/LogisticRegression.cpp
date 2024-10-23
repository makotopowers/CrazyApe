//======================================================================
// Implementation of the Logistic Regression class
// Makoto Powers
//======================================================================

//============================================================================
// INCLUDES
#include "LogisticRegression.hpp"

//============================================================================

namespace Modelling {
namespace Classification {

LogisticRegression::LogisticRegression() {}

LogisticRegression::~LogisticRegression() {}

Eigen::VectorXd LogisticRegression::fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, const double lambda, int iteration_method) {
  // @brief Fit the logistic regression model
  //
  // @param X: n x p matrix of features
  // @param y: n x m matrix of class labels
  // @param lambda: regularization parameter
  // @param iteration_method: method to use for optimization
  //
  // @return beta: p x 1 vector of coefficients

  int n = X.rows();
  int p = X.cols();

  Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd beta_old = Eigen::VectorXd::Zero(p);

  return beta;
}

Eigen::VectorXd LogisticRegression::coordinate_descent() {};

Eigen::VectorXd LogisticRegression::fixed_hessian() {};

}  // namespace Classification
}  // namespace Modelling