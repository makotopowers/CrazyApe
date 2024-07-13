#include <gtest/gtest.h>

#include "OLS.hpp"
#include "lasso.hpp"
#include "ridgeRegression.hpp"

TEST(ridgeRegressionTest, DimensionTestSingleLambda) {
  Modelling::Regression::RidgeRegression rr;

  for (int i = 0; i < 10; i++) {
    rr.fit(Eigen::MatrixXd::Random(10, 5), Eigen::VectorXd::Random(10), 0.1);
    std::vector<Eigen::VectorXd> beta = rr.get_betas();
    ASSERT_EQ(beta.size(), 1);
    for (auto x : beta) {
      ASSERT_EQ(x.size(), 5);
    }
  }
}
TEST(ridgeRegressionTest, DimensionTestMultipleLambdas) {
  Modelling::Regression::RidgeRegression rr;

  for (int i = 0; i < 10; i++) {
    rr.fit(Eigen::MatrixXd::Random(10, 5), Eigen::VectorXd::Random(10), std::vector<double>{0.1, 0.2, 0.3});
    std::vector<Eigen::VectorXd> beta = rr.get_betas();
    ASSERT_EQ(beta.size(), 3);
    for (auto x : beta) {
      ASSERT_EQ(x.size(), 5);
    }
  }
}

TEST(OLSTest, DimensionTest) {
  Modelling::Regression::OLS ols;
  for (int i = 0; i < 10; i++) {
    ols.fit(Eigen::MatrixXd::Random(10, 5), Eigen::VectorXd::Random(10));
    Eigen::VectorXd beta = ols.get_beta();
    ASSERT_EQ(beta.size(), 5);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}