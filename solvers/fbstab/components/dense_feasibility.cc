#define EIGEN_RUNTIME_NO_MALLOC
#include "drake/solvers/fbstab/components/dense_feasibility.h"

#include <cmath>

#include "drake/solvers/fbstab/linalg/static_matrix.h"
#include "drake/solvers/fbstab/components/dense_variable.h"
#include "drake/solvers/fbstab/components/dense_data.h"

namespace drake {
namespace solvers {
namespace fbstab {

DenseFeasibility::DenseFeasibility(int n, int q){
	#ifdef EIGEN_RUNTIME_NO_MALLOC
	Eigen::internal::set_is_malloc_allowed(true);
	#endif

	n_ = n;
	q_ = q;
	z1_.resize(n_);
	v1_.resize(q_);

	#ifdef EIGEN_RUNTIME_NO_MALLOC
	Eigen::internal::set_is_malloc_allowed(false);
	#endif
}

void DenseFeasibility::LinkData(DenseData *data){
	data_ = data;
}

// Check for dual infeasibility i.e.,
// max(Az) <= 0 and f'*z < 0 and ||Hz|| <= tol * ||z||
// and primal infeasibility
// v'*b < 0 and ||A'*v|| \leq tol * ||v||
void DenseFeasibility::ComputeFeasibility(const DenseVariable &x, double tol){
	// References to make the expressions cleaner
	const Eigen::MatrixXd& H = data_->H_;
	const Eigen::MatrixXd& A = data_->A_;
	const Eigen::VectorXd& f = data_->f_;
	const Eigen::VectorXd& b = data_->b_;

	// max(Az)
	v1_ = A * x.z();
	double d1 = v1_.maxCoeff();

	// f'*z
	double d2 = f.dot(x.z());

	// ||Hz||_inf
	double w = x.z().lpNorm<Eigen::Infinity>();
	z1_ = H*x.z();
	double d3 = z1_.lpNorm<Eigen::Infinity>();

	if( (d1 <=0) && (d2 < 0) && (d3 <= tol*w) ){
		dual_feasible_ = false;
	}

	// v'*b
	double p1 = b.dot(x.v());

	// ||A'v||_inf
	z1_ = A.transpose()*x.v();
	double p2 = z1_.lpNorm<Eigen::Infinity>();
	double u = x.v().lpNorm<Eigen::Infinity>();

	if( (p1 < 0) && (p2 <= tol*u) ){
		primal_feasible_ = false;
	}
}

bool DenseFeasibility::IsDualFeasible(){
	return dual_feasible_;
}
bool DenseFeasibility::IsPrimalFeasible(){
	return primal_feasible_;
}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake

