#include "ridgeRegression.hpp"

namespace Modelling {

RidgeRegression::RidgeRegression() {
  std::cout << "RidgeRegression object created" << std::endl;
}

RidgeRegression::~RidgeRegression() {
  std::cout << "RidgeRegression object destroyed" << std::endl;
}

void RidgeRegression::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, double lambda) {
  this->X = X;
  this->y = y;
  this->lambda = lambda;

  // Compute beta
  Eigen::MatrixXd XtX = X.transpose() * X;
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(XtX.rows(), XtX.cols());
  beta = (XtX + lambda * I).ldlt().solve(X.transpose() * y);
}

void RidgeRegression::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const vector<double>& lambdas) {
  this->X = X;
  this->y = y;
  this->lambdas = lambdas;

  // Compute beta
  Eigen::MatrixXd XtX = X.transpose() * X;
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(XtX.rows(), XtX.cols());

  betas = vector<Eigen::VectorXd>(lambdas.size());
  assert(betas.size() == 0);

  for (int i = 0; i < lambdas.size(); i++) {
    betas.emplace_back((XtX + lambdas[i] * I).ldlt().solve(X.transpose() * y));
  }
}

Eigen::VectorXd RidgeRegression::get_beta() const {
  return beta;
}

vector<Eigen::VectorXd> RidgeRegression::get_betas() const {
  return betas;
}

}  // namespace Modelling