#include <iostream>

#include "OLS.hpp"
#include "lasso.hpp"
#include "ridgeRegression.hpp"

int main() {
  // Create a RidgeRegression object
  Modelling::Regression::RidgeRegression rr;
  rr.fit(Eigen::MatrixXd::Random(10, 5), Eigen::VectorXd::Random(10), 0.1);
  std::cout << rr.get_betas().size() << std::endl;
  std::cout << " here" << std::endl;
  for (auto x : rr.get_betas()) {
    std::cout << x.transpose() << std::endl;
  }

  rr.fit(Eigen::MatrixXd::Random(10, 5), Eigen::VectorXd::Random(10), std::vector<double>{0.1, 0.2});
  std::cout << rr.get_betas().size() << std::endl;
  for (auto x : rr.get_betas()) {
    std::cout << x.transpose() << std::endl;
  }

  return EXIT_SUCCESS;
}