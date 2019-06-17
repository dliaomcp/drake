#include "drake/solvers/fbstab/components/dense_data.h"

#include <cmath>
#include <Eigen/Dense>


namespace drake {
namespace solvers {
namespace fbstab {

using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;

DenseData::DenseData(const MatrixXd& H,const VectorXd& f, const MatrixXd& A,const VectorXd& b, DenseQPsize size)
	: H_(H), f_(f), A_(A), b_(b){
	n_ = size.n;
	q_ = size.q;
}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake

