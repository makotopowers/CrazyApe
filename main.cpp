#include <iostream>

#include <fstream>
#include "OLS.hpp"
#include "PCA.hpp"
#include "RSS.hpp"
#include "configReader.hpp"
#include "lasso.hpp"
#include "makeData.hpp"
#include "ridgeRegression.hpp"
#include "utilities.hpp"

int main() {

  std::string confDir = "../conf/";
  std::string confFile = "global.conf";
  Tools::ConfigReader configReader;
  configReader.readConfigFile(confDir + confFile);

  Tools::Utilities utilities;
  utilities.setDebug(configReader.returnConfigValue<int>("DEBUG"));

  Data::Simulation::MakeData data;

  // set the true function
  data.set_true_function([](const Eigen::VectorXd& x) { return 2 * x(0) + 3 * x(1) + 4 * x(2) + 10; }, 3);

  // generate a synthetic dataset
  Data::Simulation::DataSet data_set = data.get_data_set(1000, 2, 0.1);

  // split the dataset into training and testing sets
  std::vector<Data::Simulation::DataSet> train_test = data.train_test_split(data_set, 0.2);

  // fit Ridge regression model with vector of lambdas
  Modelling::Regression::RidgeRegression ridge;
  ridge.fit(train_test[0].X, train_test[0].y, {0.1, 0.2, 0.5, 0.8});

  utilities.Log("Ridge regression model fit with lambdas: 0.1, 0.2, 0.5, 0.8");

  Metrics::RSS rss;

  // get prediction
  std::vector<Eigen::VectorXd> betas = ridge.get_betas();
  // Eigen::MatrixXd pred;
  // for (int i = 0; i < betas.size(); i++) {
  //   pred = ridge.predict(train_test[1].X);

  //   return EXIT_SUCCESS;
  // }
}
