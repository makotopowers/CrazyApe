//============================================================================
// PCA implementation using Eigen library
// Makoto Powers
//
// PCA can be posed as the following problem:
//
// X = U * S * V^T
//
// With X an n x p matrix, U an n x p matrix, S a p x p diagonal matrix, and V
// a p x p orthogonal matrix. The columns of U are the principal components of X,
// and the columns of V are the principal axes of X.
//============================================================================

#pragma once

//============================================================================
// INCLUDES
#include <Eigen/Dense>

//============================================================================

namespace Modelling {
namespace DimensionReduction {

class PCA {

  //   @brief PCA class
  //
  //   This class implements principal component analysis using Eigen library.
  //   The model can be fit using the fit method, and the principal components can be
  //   accessed using the get_principal_components method.

 public:
  PCA();
  ~PCA();

  void fit(const Eigen::MatrixXd& X);

  Eigen::MatrixXd get_principal_components() const;
  Eigen::VectorXd get_principal_values() const;

 private:
  Eigen::MatrixXd X;
  Eigen::MatrixXd principal_components;
  Eigen::VectorXd principal_values;
};

}  // namespace DimensionReduction
}  // namespace Modelling
