#include "drake/solvers/fbstab/components/dense_feasibility.h"

#include <cmath>

#include "drake/solvers/fbstab/linalg/static_matrix.h"
#include "drake/solvers/fbstab/components/dense_variable.h"
#include "drake/solvers/fbstab/components/dense_data.h"

namespace drake {
namespace solvers {
namespace fbstab {

DenseFeasibility::DenseFeasibility(DenseQPsize size){
	n_ = size.n;
	q_ = size.q;

	// allocate memory
	double *r1 = new double[n_];
	double *r2 = new double[n_];
	double *r3 = new double[q_];

	z1_ = StaticMatrix(r1,n_);
	z2_ = StaticMatrix(r2,n_);
	v1_ = StaticMatrix(r3,q_);
}

DenseFeasibility::~DenseFeasibility(){
	delete[] z1_.data;
	delete[] z2_.data;
	delete[] v1_.data;
}

void DenseFeasibility::LinkData(DenseData *data){
	data_ = data;
}

void DenseFeasibility::ComputeFeasibility(const DenseVariable &x, double tol){
	// check dual
	double w = x.z_.infnorm();
	// max(Az)
	v1_.gemv(x.data_->A_,x.z_);
	double d1 = v1_.max();
	// f'*z
	double d2 = StaticMatrix::dot(x.data_->f_,x.z_);
	// ||Hz||_inf
	z1_.gemv(x.data_->H_,x.z_);
	double d3 = z1_.infnorm();

	if( (d1 <=0) && (d2 < 0) && (d3 <= tol*w) ){
		dual_feasible_ = false;
	}

	// check primal 
	double u = x.v_.infnorm();
	// v'*b
	double p1 = StaticMatrix::dot(x.v_,x.data_->b_);
	// ||A'v||_inf
	z2_.gemv(x.data_->A_,x.v_,1.0,0.0,true);
	double p2 = x.z_.infnorm();
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

