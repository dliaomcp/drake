#include "drake/solvers/fbstab/dense_components/dense_feasibility.h"

#include <cmath>
#include <Eigen/Dense>

#include "drake/solvers/fbstab/dense_components/dense_variable.h"
#include "drake/solvers/fbstab/dense_components/dense_data.h"

namespace drake {
namespace solvers {
namespace fbstab {

DenseFeasibility::DenseFeasibility(int nz, int nv){
	nz_ = nz;
	nv_ = nv;
	z1_.resize(nz_);
	v1_.resize(nv_);
}

void DenseFeasibility::ComputeFeasibility(const DenseVariable &x, double tol){
	const Eigen::MatrixXd& H = data_->H();
	const Eigen::MatrixXd& A = data_->A();
	const Eigen::VectorXd& f = data_->f();
	const Eigen::VectorXd& b = data_->b();

	// The conditions for dual-infeasibility are:
	// max(Az) <= 0 and f'*z < 0 and ||Hz|| <= tol * ||z||
	v1_.noalias() = A * x.z();
	double d1 = v1_.maxCoeff();

	double d2 = f.dot(x.z());

	z1_.noalias() = H*x.z();
	double d3 = z1_.lpNorm<Eigen::Infinity>();
	double w = x.z().lpNorm<Eigen::Infinity>();

	if( (d1 <= 0) && (d2 < 0) && (d3 <= tol*w) ){
		dual_feasible_ = false;
	}

	// The conditions for primal infeasibility are:
	// v'*b < 0 and ||A'*v|| \leq tol * ||v||
	double p1 = b.dot(x.v());

	z1_.noalias() = A.transpose()*x.v();
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

