//============================================================================
// Implementation of the PCA class
// Makoto Powers
//============================================================================

//============================================================================
// INCLUDES
#include "PCA.hpp"

//============================================================================

namespace Modelling {
namespace DimensionReduction {

PCA::PCA() {}

PCA::~PCA() {}

void PCA::fit(const Eigen::MatrixXd& X_p) {
  // @brief Fit the PCA model
  //
  // @param X_p: n x p matrix of features

  // center the data
  Eigen::VectorXd mean = X_p.colwise().mean();
  Eigen::MatrixXd X_centered = X_p.rowwise() - mean.transpose();

  // compute the covariance matrix
  Eigen::MatrixXd covariance = (X_centered.transpose() * X_centered) / X_centered.rows();

  // compute the eigenvectors and eigenvalues
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(covariance);
  Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();
  Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();

  this->X = X_p;
  this->principal_components = eigenvectors;
  this->principal_values = eigenvalues;
}

Eigen::MatrixXd PCA::get_principal_components() const {
  // @brief Get the principal components
  //
  // @return principal_components: p x p matrix of principal components

  return principal_components;
}

Eigen::VectorXd PCA::get_principal_values() const {
  // @brief Get the principal values
  //
  // @return principal_values: p x 1 vector of principal values

  return principal_values;
}

}  // namespace DimensionReduction
}  // namespace Modelling