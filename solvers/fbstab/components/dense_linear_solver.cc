#include "drake/solvers/fbstab/components/dense_linear_solver.h"

#include <cmath>

#include "drake/solvers/fbstab/linalg/static_matrix.h"
#include "drake/solvers/fbstab/components/dense_variable.h"
#include "drake/solvers/fbstab/components/dense_residual.h"
#include "drake/solvers/fbstab/components/dense_data.h"

namespace drake {
namespace solvers {
namespace fbstab {

DenseLinearSolver::DenseLinearSolver(DenseQPsize size){
	n_ = size.n;
	q_ = size.q;

	// fb derivatives
	double *a1 = new double[q_];
	double *a2 = new double[q_];
	double *a3 = new double[q_];
	gamma_ = StaticMatrix(a1,q_);
	mus_ = StaticMatrix(a2,q_);
	Gamma_ = StaticMatrix(a3,q_);

	// workspace residuals
	double *b1 = new double[n_];
	double *b2 = new double[q_];
	r1_ = StaticMatrix(b1,n_);
	r2_ = StaticMatrix(b2,q_);

	// workspace hessian
	double *c = new double[n_*n_];
	K_ = StaticMatrix(c,n_,n_);
}

DenseLinearSolver::~DenseLinearSolver(){
	delete[] gamma_.data;
	delete[] mus_.data;
	delete[] Gamma_.data;

	delete[] r1_.data;
	delete[] r2_.data;

	delete[] K_.data;
}

void DenseLinearSolver::LinkData(DenseData *data){
	data_ = data;
}

void DenseLinearSolver::SetAlpha(double alpha){
	alpha_ = alpha;
}

bool DenseLinearSolver::Factor(const DenseVariable &x,const DenseVariable &xbar, double sigma){

	// compute K = H + sigma I
	K_.copy(data_->H_);
	K_.AddDiag(sigma);
	
	// K <- K + A'*diag(Gamma(x))*A
	Point2D tmp;
	for(int i = 0;i<q_;i++){
		double ys = x.y_(i) + sigma*(x.v_(i) - xbar.v_(i));
		tmp = PFBGradient(ys,x.v_(i),sigma);

		gamma_(i) = tmp.x;
		mus_(i) = tmp.y + sigma*tmp.x;
		Gamma_(i) = gamma_(i)/mus_(i);
	}
	K_.gram(data_->A_,Gamma_);

	// K = LL' in place
	int chol_flag = K_.llt();

	if(chol_flag == 0){ return true; }
	else{ return false; }
}

bool DenseLinearSolver::Solve(const DenseResidual &r, DenseVariable *x){
	// solve the system
	// K z = rz - A'*diag(1/mus)*rv
	// diag(mus) v = rv + diag(gamma)*A*z

	for(int i = 0;i<q_;i++){
		mus_(i) = 1.0/mus_(i);
	}

	r1_.copy(r.z_);
	r2_.copy(r.v_);
	// r2 = rv./mus
	r2_.RowScale(mus_);
	// r1 = -A'*r2 + r1
	r1_.gemv(data_->A_,r2_,-1.0,1.0,true); 

	// solve K z = r1
	r1_.CholSolve(K_);
	(x->z_).copy(r1_);

	// r2 = rv + diag(gamma)*A*z
	r2_.gemv(data_->A_,x->z_,1.0);
	r2_.RowScale(gamma_);
	r2_.axpy(r.v_,1.0);

	// v = diag(1/mus)*r2
	r2_.RowScale(mus_);
	(x->v_).copy(r2_);

	// y = b - Az
	(x->y_).copy(data_->b_);
	(x->y_).gemv(data_->A_,x->z_,-1.0,1.0);

	return true;
}

DenseLinearSolver::Point2D DenseLinearSolver::PFBGradient(double a,
 double b, double sigma){
	double y = 0;
	double x = 0;
	double r = sqrt(a*a + b*b);
	double d = 1.0/sqrt(2.0);

	if(r < zero_tolerance_){
		x = alpha_*(1.0-d);
		y = alpha_*(1.0-d);

	} else if((a > 0) && (b > 0)){
		x = alpha_ * (1.0- a/r) + (1.0-alpha_) * b;
		y = alpha_ * (1.0- b/r) + (1.0-alpha_) * a;

	} else {
		x = alpha_ * (1.0 - a/r);
		y = alpha_ * (1.0 - b/r);
	}

	Point2D output = {x, y};
	return output;
}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake

