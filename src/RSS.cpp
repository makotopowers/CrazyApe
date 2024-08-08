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

}  // namespace Metrics
