#define EIGEN_RUNTIME_NO_MALLOC
#include "drake/solvers/fbstab/components/dense_data.h"

#include <cmath>
#include <Eigen/Dense>


namespace drake {
namespace solvers {
namespace fbstab {

using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;

DenseData::DenseData(const MatrixXd& H,const VectorXd& f, const MatrixXd& A,const VectorXd& b)
	: H_(H), f_(f), A_(A), b_(b){
	#ifdef EIGEN_RUNTIME_NO_MALLOC
	Eigen::internal::set_is_malloc_allowed(true);
	#endif

	#ifdef EIGEN_RUNTIME_NO_MALLOC
	Eigen::internal::set_is_malloc_allowed(false);
	#endif
}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake

