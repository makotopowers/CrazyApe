#include "OLS.hpp"

namespace Modelling {

namespace Regression {

OLS::OLS() {
  std::cout << "OLS object created" << std::endl;
}

OLS::~OLS() {
  std::cout << "OLS object destroyed" << std::endl;
}

void OLS::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
  this->X = X;
  this->y = y;

  // Compute beta
  beta = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd OLS::get_beta() const {
  return beta;
}

}  // namespace Regression

}  // namespace Modelling