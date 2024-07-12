#include <gtest/gtest.h>

#include "OLS.hpp"
#include "lasso.hpp"
#include "ridgeRegression.hpp"

TEST(ridgeRegressionTest, DimensionTest) {
  Modelling::RidgeRegression rr;
  rr.fit(Eigen::MatrixXd::Random(10, 5), Eigen::VectorXd::Random(10), 0.1);
  Eigen::VectorXd beta = rr.get_beta();
  ASSERT_EQ(beta.size(), 5);
}

// TEST(lassoTest, DimensionTest) {
//   Modelling::Lasso lasso;
//   lasso.fit(Eigen::MatrixXd::Random(10, 5), Eigen::VectorXd::Random(10),
//   0.1); Eigen::VectorXd beta = lasso.get_beta(); ASSERT_EQ(beta.size(), 5);
// }

TEST(OLSTest, DimensionTest) {
  Modelling::OLS ols;
  ols.fit(Eigen::MatrixXd::Random(10, 5), Eigen::VectorXd::Random(10));
  Eigen::VectorXd beta = ols.get_beta();
  ASSERT_EQ(beta.size(), 5);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}