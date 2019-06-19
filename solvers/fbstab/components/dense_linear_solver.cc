#define EIGEN_RUNTIME_NO_MALLOC 1
#include "drake/solvers/fbstab/components/dense_linear_solver.h"

#include <cmath>
#include <iostream>
#include <Eigen/Dense>

#include "drake/solvers/fbstab/components/dense_variable.h"
#include "drake/solvers/fbstab/components/dense_residual.h"
#include "drake/solvers/fbstab/components/dense_data.h"

namespace drake {
namespace solvers {
namespace fbstab {

DenseLinearSolver::DenseLinearSolver(int n, int q){
	#ifdef EIGEN_RUNTIME_NO_MALLOC
	Eigen::internal::set_is_malloc_allowed(true);
	#endif

	n_ = n;
	q_ = q;

	K_.resize(n_,n_);
	r1_.resize(n_);
	r2_.resize(q_);
	Gamma_.resize(q_);
	mus_.resize(q_);
	gamma_.resize(q_);
	B_.resize(q_,n_);

	#ifdef EIGEN_RUNTIME_NO_MALLOC
	Eigen::internal::set_is_malloc_allowed(false);
	#endif
}

void DenseLinearSolver::LinkData(DenseData *data){
	data_ = data;
}

void DenseLinearSolver::SetAlpha(double alpha){
	alpha_ = alpha;
}

bool DenseLinearSolver::Factor(const DenseVariable &x,const DenseVariable &xbar, double sigma){
	// References to make the expressions clearer
	const Eigen::MatrixXd& H = data_->H_;
	const Eigen::MatrixXd& A = data_->A_;

	// compute K <- H + sigma I
	K_ = H + sigma*Eigen::MatrixXd::Identity(n_,n_);
	
	// K <- K + A'*diag(Gamma(x))*A
	Point2D pfb_gradient;
	for(int i = 0; i < q_; i++){
		double ys = x.y()(i) + sigma*(x.v()(i) - xbar.v()(i));
		pfb_gradient = PFBGradient(ys,x.v()(i));

		gamma_(i) = pfb_gradient.x;
		mus_(i) = pfb_gradient.y + sigma*pfb_gradient.x;
		Gamma_(i) = gamma_(i)/mus_(i);
	}

	// K += A' * Gamma * A
	// The variable B is used to avoid dynamic allocation of temporaries
	B_.noalias() = Gamma_.asDiagonal()*A;
	K_.noalias() += A.transpose()*B_;

	// Factor K = LL' in place
	Eigen::LLT<Eigen::Ref<Eigen::MatrixXd> > L(K_);

	return true;
}

// Solve the system:
// KK'z = rz - A'*diag(1/mus)*rv
// diag(mus) v = rv + diag(gamma)*A*z
// Where K = cholesky((H + sigma*I + A'*Gamma*A)) is precomputed
// by the factor routine
bool DenseLinearSolver::Solve(const DenseResidual &r, DenseVariable *x){
	if(r.n_ != x->n_ || r.q_ != x->q_){
		throw std::runtime_error("In DenseLinearSolver::Solve residual and variable objects must be the same size");
	}

	// References for clarity
	const Eigen::MatrixXd& A = data_->A_;
	const Eigen::VectorXd& b = data_->b_;

	// compute rz - A'*(rv./mus) and store it in r1_
	r2_ = r.v_.cwiseQuotient(mus_);
	r1_.noalias() =  r.z_ - A.transpose()*r2_;

	// Solve KK'*z = rz - A'*(rv./mus)
	// Where K = cholesky(H + sigma*I + A'*Gamma*A)
	// is assumed to have been computed during the factor phase
	x->z() = r1_;
	CholeskySolve(K_,x->z_);

	// Compute v = diag(1/mus) * (rv + diag(gamma)*A*z)
	// written so as to avoid temporary creation
	r2_.noalias() = A*x->z();
	r2_.noalias() = gamma_.asDiagonal()*r2_;
	r2_ += r.v_;

	// v = diag(1/mus)*r2
	x->v() = r2_.cwiseQuotient(mus_);

	// y = b - Az
	x->y() = b - A*x->z();

	return true;
}

void DenseLinearSolver::CholeskySolve(const Eigen::MatrixXd& A, Eigen::VectorXd* b){
	A.triangularView<Eigen::Lower>().solveInPlace(*b);
	A.triangularView<Eigen::Lower>().transpose().solveInPlace(*b);
}

DenseLinearSolver::Point2D DenseLinearSolver::PFBGradient(double a, double b){
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

