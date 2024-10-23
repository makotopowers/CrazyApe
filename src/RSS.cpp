//======================================================================
// Implementation of the RSS class.
// Makoto Powers
//======================================================================

//============================================================================
// INCLUDES
#include "RSS.hpp"

//============================================================================

namespace Metrics {

RSS::RSS() {}

RSS::~RSS() {}

double RSS::compute(const Eigen::VectorXd& y, const Eigen::VectorXd& y_hat) {
  // @brief Compute the residual sum of squares
  //
  // @param y: n x 1 vector of true values
  // @param y_hat: n x 1 vector of predicted values
  //
  // @return rss: residual sum of squares

  double rss = (y - y_hat).squaredNorm();

  return rss;
}

std::vector<double> RSS::compute(const Eigen::VectorXd& y, const Eigen::MatrixXd& y_hat) {
  // @brief Compute the residual sum of squares
  //
  // @param y: n x 1 vector of true values
  // @param y_hat: n x m matrix of predicted values
  //
  // @return rss: vector of residual sum of squares

  std::vector<double> rss(y_hat.cols());

  for (int i = 0; i < y_hat.cols(); i++) {
    rss[i] = (y - y_hat.col(i)).squaredNorm();
  }

  return rss;
}

}  // namespace Metrics
