//======================================================================
// Implementation of the RidgeRegression class
// Makoto Powers
//======================================================================

//============================================================================
// INCLUDES
#include "ridgeRegression.hpp"

//============================================================================

namespace Modelling {
namespace Regression {

RidgeRegression::RidgeRegression() {}

RidgeRegression::~RidgeRegression() {}

void RidgeRegression::fit(const Eigen::MatrixXd& X_p, const Eigen::VectorXd& y_p, const double lambda_p) {
  // @brief Fit the ridge regression model with a single lambda
  //
  // @param X_p: n x p matrix of features
  // @param y_p: n x 1 vector of target values
  // @param lambda_p: regularization parameter

  Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(X_p.rows(), 1);
  Eigen::MatrixXd X_intercept = Eigen::MatrixXd(X_p.rows(), X_p.cols() + 1);
  X_intercept.block(0, 0, X_p.rows(), 1) = ones;
  X_intercept.block(0, 1, X_p.rows(), X_p.cols()) = X_p;

  this->X = X_intercept;
  this->y = y_p;
  this->lambdas = std::vector<double>(1, lambda_p);
  this->betas = std::vector<Eigen::VectorXd>(lambdas.size());

  Eigen::MatrixXd XtX = X.transpose() * X;
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(XtX.rows(), XtX.cols());

  betas[0] = ((XtX + lambdas[0] * I).ldlt().solve(X.transpose() * y));

  return;
}

void RidgeRegression::fit(const Eigen::MatrixXd& X_p, const Eigen::VectorXd& y_p, const std::vector<double>& lambdas_p) {
  // @brief Fit the ridge regression model with multiple lambdas
  //
  // @param X_p: n x p matrix of features
  // @param y_p: n x 1 vector of target values
  // @param lambdas_p: vector of regularization parameters

  Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(X_p.rows(), 1);
  Eigen::MatrixXd X_intercept = Eigen::MatrixXd(X_p.rows(), X_p.cols() + 1);
  X_intercept.block(0, 0, X_p.rows(), 1) = ones;
  X_intercept.block(0, 1, X_p.rows(), X_p.cols()) = X_p;

  this->X = X_intercept;
  this->y = y_p;
  this->lambdas = lambdas_p;
  this->betas = std::vector<Eigen::VectorXd>(lambdas.size());

  Eigen::MatrixXd XtX = X.transpose() * X;
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(XtX.rows(), XtX.cols());

  for (int i = 0; i < lambdas.size(); i++) {
    betas[i] = ((XtX + lambdas[i] * I).ldlt().solve(X.transpose() * y));
  }

  return;
}

std::vector<Eigen::VectorXd> RidgeRegression::get_betas() const {
  // @brief Get the coefficients of the model
  //
  // @return vector of beta coefficients

  return betas;
}

}  // namespace Regression
}  // namespace Modelling