/*

calculate the sample covariance matrix of a data set. The sample covariance
matrix is defined as:

S = (1 / (n - 1)) * (X - mu)^T * (X - mu)

where X is an n x p matrix of data points, mu is the mean of the data set, and
the division by (n - 1) is used to correct for bias in the estimation of the
population covariance matrix.




*/

#pragma once
#include <Eigen/Dense>
#include <iostream>

namespace Statistics {

class Covariance {
  /*
  @brief Covariance class

  This class implements the sample covariance matrix using Eigen library.

  @param X: n x p matrix of data points

  */

 public:
  Covariance();
  ~Covariance();
  void fit(const Eigen::MatrixXd& X);
  Eigen::MatrixXd get_covariance() const;
}

}  // namespace Statistics