
// Ridge Regression implementation using Eigen library
// Makoto Powers

// Ridge regression can be posed as the following problem:

// beta* = argmin ||y - X * beta||^2 + lambda * ||beta||^2

// With X an n x p matrix, y an n x 1 vector, beta a p x 1 vector, and lambda a
// scalar. lambda can be seen as a regularization parameter. When lambda is 0, the
// problem is equivalent to OLS.

// In Bayesian terms, the ridge regression problem is equivalent to finding the
// mode of the posterior distribution of beta, with a Gaussian prior on beta with
// mean 0 and covariance matrix (1/lambda) * I.

// The solution to the ridge regression problem is given by:

// beta* = (X^T * X + lambda * I)^-1 * X^T * y

// where I is the identity matrix of size p x p. We compute this solution using
// Cholesky decomposition. Since our target use case is data, we can assume that
// X^T * X is real, allowing us to use LDLT decomposition. Then with

// X^T * X = LDL^T

// We have

// beta* = (X^T * X + lambda * I)^-1 * X^T * y = (LDL^T + lambda * I)^-1 * X^T * y

// and we can solve the standard linear system

// (LDL^T + lambda * I) * beta* = X^T * y

// for beta* using forward and backward substitution.

#pragma once
#include <Eigen/Dense>
#include <iostream>

namespace Modelling {

class RidgeRegression {

  //   @brief Ridge Regression class

  //   This class implements ridge regression using Eigen library. Ridge regression
  //   is a linear regression model with L2 regularization. The model can be fit
  //   using the fit method, and the coefficients can be accessed using the get_beta
  //   method.

  //   @param X: n x p matrix of features
  //   @param y: n x 1 vector of target values
  //   @param lambda: regularization parameter

 public:
  RidgeRegression();
  ~RidgeRegression();
  void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, double lambda);
  void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const vector<double>& lambdas);

  Eigen::VectorXd get_beta() const;

 private:
  Eigen::MatrixXd X;
  Eigen::VectorXd y;
  double lambda;
  Eigen::VectorXd beta;
};

}  // namespace Modelling