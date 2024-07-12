#include <iostream>

#include "OLS.hpp"
#include "lasso.hpp"
#include "ridgeRegression.hpp"

int main() {
  // Create a RidgeRegression object
  Modelling::RidgeRegression rr;
  rr.fit(Eigen::MatrixXd::Random(10, 5), Eigen::VectorXd::Random(10), 0.1);
  std::cout << "Beta: " << rr.get_beta().transpose() << std::endl;

  return EXIT_SUCCESS;
}