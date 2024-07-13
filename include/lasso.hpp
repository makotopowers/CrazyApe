//============================================================================
// Lasso implementation using Eigen library
// Makoto Powers
//
// Lasso regression can be posed as the following problem:
//
// beta* = argmin ||y - X * beta||^2 + lambda * ||beta||_1
//
// With X an n x p matrix, y an n x 1 vector, beta a p x 1 vector, and lambda a
// scalar. lambda can be seen as a regularization parameter. When lambda is 0, the
// problem is equivalent to OLS.
//
// In Bayesian terms, the lasso regression problem is equivalent to finding the
// mode of the posterior distribution of beta, with a Laplace prior on beta with
// mean 0 and scale parameter 1/lambda.
//
// Generally, there is no closed-form solution to the lasso problem. However, the
// lasso problem can be solved using proximal gradient descent. The proximal
// operator of the L1 norm is the soft-thresholding operator.

//============================================================================

#pragma once

//============================================================================
// INCLUDES
#include <Eigen/Dense>
#include <vector>

//============================================================================

namespace Modelling {
namespace Regression {

class Lasso {
  //   @brief Lasso class
  //
  //   This class implements lasso regression using Eigen library. Lasso regression
  //   is a linear regression model with L1 regularization. The model can be fit
  //   using the fit method, and the coefficients can be accessed using the get_beta
  //   method.

 public:
  Lasso();
  ~Lasso();

  void fit(const Eigen::MatrixXd& X_p, const Eigen::VectorXd& y_p, const double lambda_p);
  void fit(const Eigen::MatrixXd& X_p, const Eigen::VectorXd& y_p, const std::vector<double>& lambdas_p);

  std::vector<Eigen::VectorXd> get_betas() const;

  void set_tol(const double tol_p);
  void set_max_iter(const int max_iter_p);

 private:
  Eigen::MatrixXd X;
  Eigen::VectorXd y;
  std::vector<double> lambdas;
  std::vector<Eigen::VectorXd> betas;
  double tol;
  int max_iter;
  double alpha;

  void soft_thresholding(Eigen::VectorXd& x, const double nu);
  Eigen::VectorXd proximal_gradient_descent(const double lambda, const double tol, const int max_iter);
};

}  // namespace Regression
}  // namespace Modelling