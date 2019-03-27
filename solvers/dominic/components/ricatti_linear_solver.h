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

 	StaticMatrix tx_,tl_,tu_;

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

	double *tx_mem = new double[nx];
	double *tu_mem = new double[nu];
	double *tl_mem = new double[nx];

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

	tx_ = StaticMatrix(tx_mem,nx,1);
	tu_ = StaticMatrix(tu_mem,nu,1);
	tl_ = StaticMatrix(tl_mem,nx,1);

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

		delete[] tx_.data;
		delete[] tu_.data;
		delete[] tl_.data;
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

	// return Gamma to its initial shape
	Gamma_.reshape(nv,1);

	// TODO take some kind of action if one of the cholesky factorizations fail
	return true;
}

bool RicattiLinearSolver::Solve(const MSResidual &r, MSVariable *dx){
	int N = N_; int nx = nx_; int nu = nu_; 
	int nz = nz_; int nl = nl_; int nv = nv_;

	// compute reduced residuals 
	// r1 = rz - A'*(rv./mus)
	r1_.copy(r.z_);
	for(int i = 0;i<nv;i++){
		Gamma_(i) = r.v_(i)/mus_(i);
	}
	data_->gemvAT(Gamma_,-1.0,1.0,&r1_);
	// r2 = -rl
	r2_.copy(r.l_);
	r2_ *= -1.0;

	r1_.reshape(nx+nu,N+1);
	r2_.reshape(nx,N+1);

	// forward recursion for h and theta
	// i = 0
	th_(0).copy(r2_.col(0));
	tx_.copy(r1_.col(0).subvec(0,nx-1));

	h_(0).copy(th_(0));
	h_(0).CholSolve(L_(0));
	h_(0).axpy(tx_,-1.0);

	StaticMatrix rlp, rxp;
	for(int i = 0;i<N;i++){
		// compute theta(i+1)
		tx_.copy(h_(i)); // rx = h
		tx_.LeftCholApply(M_(i)); // rx = inv(M)*rx

		tu_.copy(r1_.col(i).subvec(nx,nx+nu-1));
		tu_.gemv(SM_(i),tx_,1.0,1.0); // ru = SM*rx + ru
		tu_.LeftCholApply(SG_(i));

		rlp = r2_.col(i+1);
		th_(i+1).copy(rlp);
		th_(i+1).gemv(P_(i),tu_,1.0,1.0); // th(i+1) += P*ru
		th_(i+1).gemv(AM_(i),tx_,1.0,1.0); // th(i+1) += AM*rx

		// compute h(i+1)
		rxp = r1_.col(i+1).subvec(0,nx-1);
		h_(i+1).copy(th_(i+1));
		h_(i+1).CholSolve(L_(i+1));
		h_(i+1).axpy(rxp,-1.0);
	}

	// compute xN,uN, and lN
	tx_.copy(h_(N));
	tx_.LeftCholApply(M_(N)); 
	tu_.copy(r1_.col(N).subvec(nx,nx+nu-1)); // uN = rx(N)
	tu_.gemv(SM_(N),tx_,1.0,1.0); // uN = SM*inv(M)*h + rx(N)
	tu_.CholSolve(SG_(N)); // uN = inv(SG*SG')uN

	tx_.copy(h_(N));
	tx_.LeftCholApply(M_(N));
	tx_.gemv(SM_(N),tu_,1.0,1.0,true); // xN = inv(M)*h + SM'*uN
	tx_ *= -1.0; 
	tx_.LeftCholApply(M_(N),true); // xN = -inv(M')*xN

	tl_.copy(tx_);
	tl_.axpy(th_(N),1.0); // lN = xN + theta(N)
	tl_ *= -1.0; 
	tl_.CholSolve(L_(N)); // lN = -inv(L*L')*lN

	// copy into solution vector
	dx->z_.reshape(nx+nu,N+1);
	dx->l_.reshape(nx,N+1);
	StaticMatrix xN = dx->z_.col(N).subvec(0,nx-1); 
	StaticMatrix uN = dx->z_.col(N).subvec(nx,nx+nu-1);
	StaticMatrix lN = dx->l_.col(N);

	xN.copy(tx_);
	uN.copy(tu_);
	lN.copy(tl_);

	// backwards recursion for the solution
	for(int i = N-1;i>=0;i--){
		// compute u(i)
		tx_.copy(h_(i));
		tx_.LeftCholApply(M_(i));

		StaticMatrix ui = dx->z_.col(i).subvec(nx,nx+nu-1);
		ui.gemv(SM_(i),tx_,1.0,0.0); // ui = SM*(inv(M)*h)

		StaticMatrix ru = r1_.col(i).subvec(nx,nx+nu-1);
		ui.axpy(ru,1.0); // ui += ru
		ui.LeftCholApply(SG_(i)); // ui = inv(SG)*ui

		StaticMatrix lip = dx->l_.col(i+1);
		ui.gemv(P_(i),lip,1.0,1.0,true); // ui += P'*lp
		ui.LeftCholApply(SG_(i),true); // ui = SG'*ui

		// compute x(i)
		StaticMatrix xi = dx->z_.col(i).subvec(0,nx-1);
		xi.copy(h_(i));
		xi.LeftCholApply(M_(i)); // x(i) = inv(M)*h(i)

		xi.gemv(SM_(i),ui,1.0,1.0,true); // x(i) += SM'*u(i)
		xi.gemv(AM_(i),lip,1.0,1.0,true); // x(i) += AM'*l(i+1)

		xi.LeftCholApply(M_(i),true); // x(i) = inv(M')*xi
		xi *= -1.0;

		// compute l(i)
		StaticMatrix li = dx->l_.col(i);
		li.copy(th_(i));
		li.axpy(xi,1.0);
		li.CholSolve(L_(i));
		li *= -1.0;
	}

	// recover ieq dual variables
	// dv = (r.v + diag(gamma) * A*dz)*diag(1/mus)
	StaticMatrix dv = dx->v_;
	dv.copy(r.v_);
	data_->gemvA(dx->z_,1.0,0.0,&Gamma_); // Gamma is being used as a temp

	for(int i = 0;i<nv;i++){
		dv(i) = (dv(i) + gamma_(i)*Gamma_(i))/mus_(i);
	}

	// compute dy for the linesearch
	// dy = b - A*dz
	StaticMatrix dy = dx->y_;
	data_->gemvA(dx->z_,-1.0,0.0,&dy);
	data_->axpyb(1.0,&dy);

	// return the solution vectors to the correct shape
	dx->z_.reshape(nz,1);
	dx->l_.reshape(nl,1);

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
