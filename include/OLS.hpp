/*
OLS implementation using Eigen library
Makoto Powers

OLS can be posed as the following problem:

beta* = argmin ||y - X * beta||^2

With X an n x p matrix, y an n x 1 vector, beta a p x 1 vector.

The solution to the OLS problem is given by:

beta* = (X^T * X)^-1 * X^T * y

We compute this solution using Cholesky decomposition. Since our target use case
is data, we can assume that X^T * X is real, allowing us to use LDLT
decomposition. Then with

X^T * X = LDL^T

We have

beta* = (X^T * X)^-1 * X^T * y = (LDL^T)^-1 * X^T * y

and we can solve the standard linear system

(LDL^T) * beta* = X^T * y

for beta* using forward and backward substitution.
*/

#pragma once
#include <Eigen/Dense>
#include <iostream>

namespace Modelling {

namespace Regression {

class OLS {
  /*
  @brief OLS class

  This class implements ordinary least squares regression using Eigen library.
  The model can be fit using the fit method, and the coefficients can be
  accessed using the get_beta method.

  @param X: n x p matrix of features
  @param y: n x 1 vector of target values
  */
 public:
  OLS();
  ~OLS();
  void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
  Eigen::VectorXd get_beta() const;

 private:
  Eigen::MatrixXd X;
  Eigen::VectorXd y;
  Eigen::VectorXd beta;
};

}  // namespace Regression

}  // namespace Modelling
