#include <gtest/gtest.h>

#include <random>
#include "OLS.hpp"
#include "lasso.hpp"
#include "makeData.hpp"
#include "ridgeRegression.hpp"

TEST(ridgeRegressionTest, DimensionTestSingleLambda) {
  Modelling::Regression::RidgeRegression rr;

  for (int i = 0; i < 10; i++) {
    rr.fit(Eigen::MatrixXd::Random(10, 5), Eigen::VectorXd::Random(10), 0.1);
    std::vector<Eigen::VectorXd> beta = rr.get_betas();
    ASSERT_EQ(beta.size(), 1);
    for (auto x : beta) {
      ASSERT_EQ(x.size(), 6);
    }
  }
}

TEST(OLSTest, accuracyTest) {
  Modelling::Regression::OLS ols;
  Data::Simulation::MakeData data;
  std::vector<double> betas;
  std::default_random_engine generator;
  int num_beta;
  int runs = 100;

  for (int i = 0; i < runs; i++) {
    // set random betas
    num_beta = rand() % 20 + 1;
    betas.clear();
    for (int j = 0; j < num_beta; j++) {
      betas.push_back(rand() % 100);
    }

    // set the true function
    data.set_true_function(
        [betas](const Eigen::VectorXd& x) {
          double y = 0;
          for (int i = 0; i < betas.size(); i++) {
            y += betas[i] * x(i);
          }
          return y;
        },
        num_beta);

    // generate a synthetic dataset
    Data::Simulation::DataSet data_set = data.get_data_set(100, rand() % 10, 0.1);

    // fit the OLS model
    ols.fit(data_set.X, data_set.y);

    // get the coefficients
    Eigen::VectorXd beta = ols.get_beta();

    for (int i = 0; i < num_beta; i++) {
      ASSERT_NEAR(beta(i + 1), betas[i], 0.1);
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
      ASSERT_EQ(x.size(), 6);
    }
  }
}

TEST(OLSTest, DimensionTest) {
  Modelling::Regression::OLS ols;
  for (int i = 0; i < 10; i++) {
    ols.fit(Eigen::MatrixXd::Random(10, 5), Eigen::VectorXd::Random(10));
    Eigen::VectorXd beta = ols.get_beta();
    ASSERT_EQ(beta.size(), 6);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}