#include "ridgeRegression.hpp"

namespace Modelling {

RidgeRegression::RidgeRegression() {
  std::cout << "RidgeRegression object created" << std::endl;
}

RidgeRegression::~RidgeRegression() {
  std::cout << "RidgeRegression object destroyed" << std::endl;
}

void RidgeRegression::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
                          double lambda) {
  this->X = X;
  this->y = y;
  this->lambda = lambda;

  // Compute beta
  Eigen::MatrixXd XtX = X.transpose() * X;
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(XtX.rows(), XtX.cols());
  beta = (XtX + lambda * I).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd RidgeRegression::get_beta() const { return beta; }

}  // namespace Modelling