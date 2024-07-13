//======================================================================
// Implementation of the Lasso class
// Makoto Powers
// Reference:
// https://www.stats.ox.ac.uk/~rebeschi/teaching/AFoL/22/material/lecture13.pdf
//======================================================================

//============================================================================
// INCLUDES
#include "lasso.hpp"

//============================================================================

namespace Modelling {
namespace Regression {

Lasso::Lasso() : tol(1e-4), max_iter(1000), alpha(0.01) {};

Lasso::~Lasso() {}

void Lasso::fit(const Eigen::MatrixXd& X_p, const Eigen::VectorXd& y_p, const double lambda_p) {
  // @brief Fit the lasso regression model with a single lambda
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

  betas[0] = proximal_gradient_descent(lambda_p, tol, max_iter);

  return;
}

void Lasso::fit(const Eigen::MatrixXd& X_p, const Eigen::VectorXd& y_p, const std::vector<double>& lambdas_p) {
  // @brief Fit the lasso regression model with multiple lambdas
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

  for (int i = 0; i < lambdas.size(); i++) {
    betas[i] = proximal_gradient_descent(lambdas[i], tol, max_iter);
  }

  return;
}

std::vector<Eigen::VectorXd> Lasso::get_betas() const {
  return betas;
}

void Lasso::set_tol(const double tol_p) {
  this->tol = tol_p;
}

void Lasso::set_max_iter(const int max_iter_p) {
  this->max_iter = max_iter_p;
}

void Lasso::soft_thresholding(Eigen::VectorXd& x, const double nu) {

  for (int i = 0; i < x.size(); i++) {
    if (x(i) > nu) {
      x(i) -= nu;
    } else if (x(i) < -nu) {
      x(i) += nu;
    } else {
      x(i) = 0;
    }
  }

  return;
}

Eigen::VectorXd Lasso::proximal_gradient_descent(const double lambda, const double tol, const int max_iter) {
  // @brief Perform proximal gradient descent to solve the lasso problem
  //
  // @param lambda: regularization parameter
  // @param tol: tolerance for convergence
  // @param max_iter: maximum number of iterations
  //
  // @return beta: coefficients of the model

  Eigen::VectorXd beta = (X.transpose() * X).ldlt().solve(X.transpose() * y);
  Eigen::VectorXd beta_new = beta;
  Eigen::VectorXd grad_f;

  // t should be the max eigenvalue of 1/n *X^T * X
  const double alpha = ((1.0 / (X.rows())) * X.transpose() * X).eigenvalues().real().cwiseAbs().maxCoeff();

  for (int i = 0; i < max_iter; i++) {
    beta = beta_new;
    grad_f = X.transpose() * (X * beta - y) / X.rows();
    beta_new = beta - alpha * grad_f;
    soft_thresholding(beta_new, lambda * alpha);
    if ((beta_new - beta).norm() < tol) {
      break;
    }
  }

  return beta_new;
}

}  // namespace Regression
}  // namespace Modelling
