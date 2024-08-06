#include <iostream>

#include "OLS.hpp"
#include "PCA.hpp"
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

  // create data object

  //running
  Data::Simulation::MakeData data;

  // set the true function
  data.set_true_function([](const Eigen::VectorXd& x) { return 2 * x(0) + 3 * x(1) + 10; }, 2);

  // generate a synthetic dataset
  Data::Simulation::DataSet data_set = data.get_data_set(100, 0, 0.1);

  // create an OLS object
  Modelling::Regression::OLS ols;

  // fit the OLS model
  ols.fit(data_set.X, data_set.y);

  // get the coefficients
  Eigen::VectorXd beta = ols.get_beta();

  std::cout << "OLS coefficients: " << beta.transpose() << std::endl;

  // create a ridge regression object
  Modelling::Regression::RidgeRegression ridge;

  // fit the ridge regression model
  ridge.fit(data_set.X, data_set.y, 0.1);

  // get the coefficients
  std::vector<Eigen::VectorXd> betas = ridge.get_betas();

  for (int i = 0; i < betas.size(); i++) {
    std::cout << "Ridge coefficients: " << betas[i].transpose() << std::endl;
  }

  //   // create a lasso regression object
  Modelling::Regression::Lasso lasso;

  // fit the lasso regression model
  lasso.fit(data_set.X, data_set.y, 0.0);

  // get the coefficients
  betas = lasso.get_betas();

  for (int i = 0; i < betas.size(); i++) {
    std::cout << "Lasso coefficients: " << betas[i].transpose() << std::endl;
  }

  // create a PCA object
  Modelling::DimensionReduction::PCA pca;

  // fit the PCA model
  pca.fit(data_set.X);

  // get the explained variance ratio
  Eigen::MatrixXd principal_components = pca.get_principal_components();

  std::cout << "Principal components: " << principal_components.transpose() << std::endl;

  return EXIT_SUCCESS;
}