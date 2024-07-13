//============================================================================
// Implementation of the OLS class
// Makoto Powers
//============================================================================

//============================================================================
// INCLUDES
#include "OLS.hpp"

//============================================================================

namespace Modelling {
namespace Regression {

OLS::OLS() {}

OLS::~OLS() {}

void OLS::fit(const Eigen::MatrixXd& X_p, const Eigen::VectorXd& y_p) {
  // @brief Fit the OLS model
  //
  // @param X_p: n x p matrix of features
  // @param y_p: n x 1 vector of target values

  // add column of ones to X for the intercept
  Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(X_p.rows(), 1);
  Eigen::MatrixXd X_intercept = Eigen::MatrixXd(X_p.rows(), X_p.cols() + 1);
  X_intercept.block(0, 0, X_p.rows(), 1) = ones;
  X_intercept.block(0, 1, X_p.rows(), X_p.cols()) = X_p;

  this->X = X_intercept;
  this->y = y_p;

  // Compute beta
  this->beta = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd OLS::get_beta() const {
  // @brief Get the beta coefficients
  //
  // @return beta: p x 1 vector of coefficients

  return beta;
}

}  // namespace Regression
}  // namespace Modelling