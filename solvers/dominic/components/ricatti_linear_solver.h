#pragma once

#include <cmath>

#include "drake/solvers/dominic/linalg/static_matrix.h"
#include "drake/solvers/dominic/linalg/matrix_sequence.h"
#include "drake/solvers/dominic/components/mpc_data.h"
#include "drake/solvers/dominic/components/mpc_variable.h"
#include "drake/solvers/dominic/components/mpc_residual.h"

namespace drake {
namespace solvers {
namespace fbstab {


class RicattiLinearSolver{
 public:
 	struct Point2D {double x; double y;};

 	RicattiLinearSolver(QPsizeMPC size);
 	~RicattiLinearSolver();

 	void LinkData(MPCData *data);
 	bool Factor(const MSVariable &x, const MSVariable &xbar, double sigma);
 	bool Solve(const MSResidual &r, MSVariable *dx);

 	double alpha_ = 0.95;

 	MatrixSequence Q_, R_, S_;
 	MatrixSequence P_;
 	MatrixSequence SG_;
 	MatrixSequence M_;
 	MatrixSequence L_;
 	MatrixSequence SM_;
 	MatrixSequence AM_;

 	MatrixSequence h_;
 	MatrixSequence th_;

 	StaticMatrix gamma_; 
 	StaticMatrix mus_;
 	StaticMatrix Gamma_;

 	StaticMatrix Etemp_;
 	StaticMatrix Linv_;
 	StaticMatrix r1_;
 	StaticMatrix r2_;

 	StaticMatrix rx_,rl_,ru_,rxp_,rlp_;

 private:
 	int N_, nx_, nu_, nc_;
 	int nz_, nl_, nv_;

 	MPCData *data_ = nullptr;
 	double zero_tol_ = 1e-13;
 	bool memory_allocated_ = false;

 	Point2D PFBgrad(double a, double b, double sigma);
};

RicattiLinearSolver::RicattiLinearSolver(QPsizeMPC size){
	// compute sizes
	N_ = size.N;
	nx_ = size.nx;
	nu_ = size.nu;
	nc_ = size.nc;

	nz_ = (N_+1)*(nx_+nu_);
	nl_ = (N_+1)*(nx_);
	nv_ = (N_+1)*(nc_);

	int N = N_; int nx = nx_; int nu = nu_; int nc = nc_;
	int nz = nz_; int nl = nl_; int nv = nv_;

	int NN = N+1;

	double *Q_mem = new double[nx*nx*NN];
	double *S_mem = new double[nu*nx*NN];
	double *R_mem = new double[nu*nu*NN];

	double *P_mem = new double[nx*nu*NN];
	double *SG_mem = new double[nu*nu*NN];
	double *M_mem = new double[nx*nx*NN];
	double *L_mem = new double[nx*nx*NN];
	double *SM_mem = new double[nu*nx*NN];
	double *AM_mem = new double[nx*nx*NN];

	double *h_mem = new double[nx*1*NN];
	double *th_mem = new double[nx*1*NN];

	double *gamma_mem = new double[nv];
	double *Gamma_mem = new double[nv];
	double *mus_mem = new double[nv];

	double *Etemp_mem = new double[nc*nx];

	double *Linv_mem = new double[nx*nx];
	double *r1_mem = new double[nz];
	double *r2_mem = new double[nl];

	double *rx_mem = new double[nx];
	double *ru_mem = new double[nu];
	double *rl_mem = new double[nx];
	double *rxp_mem = new double[nx];
	double *rlp_mem = new double[nx];

	Q_ = MatrixSequence(Q_mem,N+1,nx,nx);
	S_ = MatrixSequence(S_mem,N+1,nu,nx);
	R_ = MatrixSequence(R_mem,N+1,nu,nu);

	P_ = MatrixSequence(P_mem,N+1,nx,nu);
	SG_ = MatrixSequence(SG_mem,N+1,nu,nu);
	M_ = MatrixSequence(M_mem,N+1,nx,nx);
	L_ = MatrixSequence(L_mem,N+1,nx,nx);
	SM_ = MatrixSequence(SM_mem,N+1,nu,nx);
	AM_ = MatrixSequence(AM_mem,N+1,nx,nx);

	h_ = MatrixSequence(h_mem,N+1,nx,1);
	th_ = MatrixSequence(th_mem,N+1,nx,1);

	gamma_ = StaticMatrix(gamma_mem,nv,1);
	mus_ = StaticMatrix(mus_mem,nv,1);
	Gamma_ = StaticMatrix(Gamma_mem,nv,1);

	Etemp_ = StaticMatrix(Etemp_mem,nc,nx);

	Linv_ = StaticMatrix(Linv_mem,nx,nx);
	r1_ = StaticMatrix(r1_mem,nz,1);
	r2_ = StaticMatrix(r2_mem,nl,1);

	rx_ = StaticMatrix(rx_mem,nx,1);
	ru_ = StaticMatrix(ru_mem,nu,1);
	rl_ = StaticMatrix(rl_mem,nx,1);
	rxp_ = StaticMatrix(rxp_mem,nx,1);
	rlp_ = StaticMatrix(rlp_mem,nx,1);

	memory_allocated_ = true;
}

RicattiLinearSolver::~RicattiLinearSolver(){
	if(memory_allocated_){
		Q_.DeleteMemory();
		R_.DeleteMemory();
		S_.DeleteMemory();

		P_.DeleteMemory();
		SG_.DeleteMemory();
		M_.DeleteMemory();
		L_.DeleteMemory();
		SM_.DeleteMemory();
		AM_.DeleteMemory();

		h_.DeleteMemory();
		th_.DeleteMemory();

		delete[] gamma_.data;
		delete[] mus_.data;
		delete[] Gamma_.data;

		delete[] Etemp_.data;

		delete[] Linv_.data;
		delete[] r1_.data;
		delete[] r2_.data;

		delete[] rx_.data;
		delete[] ru_.data;
		delete[] rl_.data;
		delete[] rxp_.data;
		delete[] rlp_.data;
	}
}

void RicattiLinearSolver::LinkData(MPCData *data){
	data_ = data;
}

bool RicattiLinearSolver::Factor(const MSVariable&x, const MSVariable &xbar, double sigma){
	int N = N_; int nc = nc_; int nv = nv_;

	if(data_ == nullptr)
		throw std::runtime_error("Data not linked in MSVariable");
	
	// compute the barrier vector Gamma= gamma/mus
	Point2D tmp;
	for(int i = 0;i<nv;i++){
		double ys = x.y_(i) + sigma*(x.v_(i) - xbar.v_(i));
		tmp = PFBgrad(ys,x.v_(i),sigma);

		gamma_(i) = tmp.x;
		mus_(i) = tmp.y + sigma*tmp.x;
		Gamma_(i) = gamma_(i)/mus_(i);
	}
	Gamma_.reshape(nc,N+1);

	// augment the Hessians with the barrier terms
	Q_.copy(data_->Q_);
	S_.copy(data_->S_);
	R_.copy(data_->R_);
	StaticMatrix Qi,Ri,Si,Ei,Li;
	for(int i=0;i<=N;i++){
		Qi = Q_(i);
		Ri = R_(i);
		Si = S_(i);
		Ei = data_->E_(i);
		Li = data_->L_(i);
		
		// Q,R += sigma*I
		Qi.AddDiag(sigma);
		Ri.AddDiag(sigma);

		// Add barriers associated with E(i)x(i) + L(i)u(i) + d() <=0
		Qi.gram(Ei,Gamma_.col(i)); // Q += E(i)'*diag(Gamma(i))*E(i)
		Ri.gram(Li,Gamma_.col(i)); // R += L(i)'*diag(Gamma(i))*L(i)

		// S(i) += L(i) ' * diag(Gamma(i)) * E(i)
		Etemp_.copy(data_->E_(i));
		Etemp_.RowScale(Gamma_.col(i));
		Si.gemm(Li,Etemp_,1.0,1.0,true,false);
	}

	// begin the ricatti recursion
	// base case, Pi = sigma I, L = chol(Pi)
	L_(0).eye(sqrt(sigma));
	for(int i = 0;i<N;i++){
		// get inv(L(i))
		Linv_.eye();
		Linv_.RightCholApply(L_(i)); // inv(L) = I*inv(L)
		Linv_.tril(); // clear the upper triangle

		// compute QQ = Q+inv(L*L') = Q + inv(L)'*inv(L)
		// then factor M = chol(QQ)
		M_(i).copy(Q_(i));
		M_(i).gram(Linv_);
		M_(i).llt();

		// compute AM = A*inv(M)' and SM = S*inv(M)'
		AM_(i).copy(data_->A_(i));
		AM_(i).RightCholApply(M_(i),true);

		SM_(i).copy(S_(i));
		SM_(i).RightCholApply(M_(i),true);

		// compute SG = chol(R - SM*SM')
		SG_(i).copy(R_(i));
		SG_(i).gram(SM_(i),-1.0,true);
		SG_(i).llt();

		// compute P = (A*inv(QQ)S' - B)*inv(SG)'
		// P = (AM*SM' - B)*inv(SG)'
		P_(i).copy(data_->B_(i));
		P_(i) *= -1.0;
		P_(i).gemm(AM_(i),SM_(i),1.0,1.0,false,true);
		P_(i).RightCholApply(SG_(i),true);

		// compute L(i+1) = chol(Pi(i+1))
		// Pi(i+1) = P*P' + AM*AM' + sigma I
		L_(i+1).eye(sigma);
		L_(i+1).gram(P_(i),1.0,true);
		L_(i+1).gram(AM_(i),1.0,true);
		L_(i+1).llt();
	}

	// finish the recursion
	Linv_.eye();
	Linv_.RightCholApply(L_(N)); // inv(L) = I*inv(L)
	Linv_.tril(); // clear the upper triangle

	// compute QQ = Q+inv(L*L') 
	// then factor M = chol(QQ)
	M_(N).copy(Q_(N));
	M_(N).gram(Linv_);
	M_(N).llt();

	SM_(N).copy(S_(N));
	SM_(N).RightCholApply(M_(N),true);

	SG_(N).copy(R_(N));
	SG_(N).gram(SM_(N),-1.0,true);
	SG_(N).llt();

	// TODO take some kind of action if one of the cholesky factorizations fail
	return true;
}

bool RicattiLinearSolver::Solve(const MSResidual &r, MSVariable *dx){
	int N = N_; int nc = nc_; int nv = nv_;


	return true;
}

RicattiLinearSolver::Point2D RicattiLinearSolver::PFBgrad(double a,
 double b, double sigma){
	double y = 0;
	double x = 0;
	double r = sqrt(a*a + b*b);

	if(r < zero_tol_){
		double d = 1.0/sqrt(2.0);
		x = alpha_*(1.0-d);
		y = alpha_*(1.0-d);

	} else if((a > 0) && (b > 0)){
		x = alpha_ * (1.0- a/r) + (1.0-alpha_) * b;
		y = alpha_ * (1.0- b/r) + (1.0-alpha_) * a;

	} else {
		x = alpha_ * (1.0 - a/r);
		y = alpha_ * (1.0 - b/r);
	}

	Point2D out = {x, y};
	return out;
}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
