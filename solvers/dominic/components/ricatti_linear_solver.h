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
 private:
 	int N_, nx_, nu_, nc_;
 	int nz_, nl_, nv_;

 	MPCData *data_ = nullptr;
 	double zero_tol_ = 1e-13;
 	bool memory_allocated_ = false;

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

 	StaticMatrix Linv_;
 	StaticMatrix r1_;
 	StaticMatrix r2_;

 	StaticMatrix rx_,rl_,ru_,rxp_,rlp_;

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

	double *Linv_mem = new double[nx*nx];
	double *r1_mem = new double[nz];
	double *r2_mem = new double[nl];

	double *rx_mem = new double[nx];
	double *ru_mem = new double[nu];
	double *rl_mem = new double[nx];
	double *rxp_mem new double[nx];
	double *rlp_mem = new double[nx];

	Q_ = MatrixSequence(Qmem,N+1,nx,nx);
	S_ = MatrixSequence(Smem,N+1,nu,nx);
	R_ = MatrixSequence(Rmem,N+1,nu,nu);

	P_ = MatrixSequence(Pmem,N+1,nx,nu);
	SG_ = MatrixSequence(SGmem,N+1,nu,nu);
	M_ = MatrixSequence(Mmem,N+1,nx,nx);
	L_ = MatrixSequence(Lmem,N+1,nx,nx);
	SM_ = MatrixSequence(SMmem,N+1,nu,nx);
	AM_ = MatrixSequence(AMmem,N+1,nx,nx);

	h_ = MatrixSequence(hmem,N+1,nx,1);
	th_ = MatrixSequence(thmem,N+1,nx,1);

	gamma_ = StaticMatrix(gamma_mem,nv,1);
	mus_ = StaticMatrix(mus_mem,nv,1);
	Gamma_ = StaticMatrix(Gamma_mem,nv,1);

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
	int N = N_; int nx = nx_; int nu = nu_; int nc = nc_;

	// compute the barrier vector Gamma= gamma/mus
	Point2D tmp;
	for(int i = 0;i<q;i++){
		double ys = x.y(i) + sigma*(x.v(i) - xbar.v(i));
		tmp = PFBgrad(ys,x.v(i),sigma);

		gamma(i) = tmp.x;
		mus(i) = tmp.y + sigma*tmp.x;
		Gamma(i) = gamma(i)/mus(i);
	}

	// augment the Hessians with the barrier terms
	Q_.copy(data->Q_);
	S_.copy(data->S_);
	R_.copy(data->R_);
	for(int i=0;i<=N;i++){
		
	}

}

bool RicattiLinearSolver::Solve(const MSResidual &r, MSVariable *dx){

}

RicattiLinearSolver::Point2D DenseLinearSolver::PFBgrad(double a,
 double b, double sigma){
	double y = 0;
	double x = 0;
	double r = sqrt(a*a + b*b);
	double d = 1.0/sqrt(2.0);

	if(r < zero_tol){
		x = alpha*(1.0-d);
		y = alpha*(1.0-d);

	} else if((a > 0) && (b > 0)){
		x = alpha * (1.0- a/r) + (1.0-alpha) * b;
		y = alpha * (1.0- b/r) + (1.0-alpha) * a;

	} else {
		x = alpha * (1.0 - a/r);
		y = alpha * (1.0 - b/r);
	}

	Point2D out = {x, y};
	return out;
}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
