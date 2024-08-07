#include <iostream>

#include <fstream>
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

  Data::Simulation::MakeData data;

  // set the true function
  data.set_true_function([](const Eigen::VectorXd& x) { return 2 * x(0) + 3 * x(1) + 10; }, 2);

  Eigen::MatrixXd cov = Eigen::MatrixXd(2, 2);
  cov << 1, 0.8, 0.8, 1;
  Eigen::VectorXd mean = Eigen::VectorXd::Zero(2);
  data.make_data_multivariate_normal(mean, cov, 100);

  // generate a synthetic dataset
  Data::Simulation::DataSet data_set = data.get_data_set(100, 0, 0.1);

  // mean = 0,0
  // cov = [[1, 0.5], [0.5, 1]]

  Eigen::MatrixXd multi = data.get_underlying_data();

  // std::cout << "Multivariate normal: " << multi.cols() << " " << multi.rows() << std::endl;

  // dump the dataset in a file
  std::ofstream file;
  file.open("../data.csv");
  file << "X1,X2\n";
  for (int i = 0; i < multi.rows(); i++) {
    file << multi(i, 0) << "," << multi(i, 1) << "\n";
  }
  file.close();

  return EXIT_SUCCESS;
}