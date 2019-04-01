#include "drake/solvers/dominic/components/mpc_data.h"

#include "drake/solvers/dominic/linalg/static_matrix.h"
#include "drake/solvers/dominic/linalg/matrix_sequence.h"


namespace drake {
namespace solvers {
namespace fbstab {


MPCData::MPCData(double **Q, double **R, double **S, double **q, 
	double **r,double **A, double **B, double **c,
 	double **E, double **L, double **d, double *x0, 
 	QPsizeMPC size){

	N_ = size.N;
	nx_ = size.nx;
	nu_ = size.nu;
	nc_ = size.nc;

	nz_ = (N_+1)*(nx_+nu_);
	nl_ = (N_+1)*nx_;
	nv_ = (N_+1)*nc_;

	int nx = nx_; int nu = nu_; int nc = nc_; int N = N_;

	// cost
	Q_ = MatrixSequence(Q,N+1,nx,nx);
	R_ = MatrixSequence(R,N+1,nu,nu);
	S_ = MatrixSequence(S,N+1,nu,nx);
	q_ = MatrixSequence(q,N+1,nx,1);
	r_ = MatrixSequence(r,N+1,nu,1);
	// dynamics
	A_ = MatrixSequence(A,N,nx,nx);
	B_ = MatrixSequence(B,N,nx,nu);
	c_ = MatrixSequence(c,N,nx,1);
	x0_ = StaticMatrix(x0,nx,1);
	// inequalities
	E_ = MatrixSequence(E,N+1,nc,nx);
	L_ = MatrixSequence(L,N+1,nc,nu);
	d_ = MatrixSequence(d,N+1,nc,1);
}

// y <- a*H*x + b*y
void MPCData::gemvH(const StaticMatrix &x, double a, double b, StaticMatrix *y){
	int nx = nx_; int nu = nu_; int N = N_;
	// TODO check sizes

	if(b == 0.0){
		y->fill(0.0);
	}else if(b != 1.0){
		y->operator*=(b);
	}

	y->reshape(nx+nu,N+1);
	StaticMatrix v = x.getreshape(nx+nu,N+1);
	
	StaticMatrix Q,S,R;
	StaticMatrix yi,yx,yu;
	StaticMatrix vi,vx,vu;
	for(int i=0;i<N+1;i++){
		Q = Q_(i);
		S = S_(i);
		R = R_(i);
		// get aliases for yx = y(1:nx,i), yu = y(nx+1:nx+nu,i)
		yi = y->col(i);
		yx = yi.subvec(0,nx-1);
		yu = yi.subvec(nx,nx+nu-1);
		// same for vx = v(1:nx,i), vu = v(nx+1:nx+nu,i)
		vi = v.col(i);
		vx = vi.subvec(0,nx-1);
		vu = vi.subvec(nx,nx+nu-1);
		// yx += a*(Q(i)*vx + S(i)'*vu)
		yx.gemv(Q,vx,a,1.0);
		yx.gemv(S,vu,a,1.0,true);
		// yu += S(i)*vx + R(i)*vu
		yu.gemv(S,vx,a,1.0);
		yu.gemv(R,vu,a,1.0);
	}

	y->reshape((N+1)*(nx+nu),1);
}

void MPCData::gemvA(const StaticMatrix &x, double a, double b, StaticMatrix *y){
	int nx = nx_; int nu = nu_; int nc = nc_; int N = N_;
	// TODO check sizes
	
	if(b == 0.0){
		y->fill(0.0);
	}else if(b != 1.0){
		y->operator*=(b);
	}

	y->reshape(nc,N+1);
	StaticMatrix z = x.getreshape(nx+nu,N+1);
	
	StaticMatrix E,L;
	StaticMatrix yi;
	StaticMatrix zi,xi,ui;
	for(int i=0;i<N+1;i++){
		E = E_(i);
		L = L_(i);
		// get aliases for yi = y(:,i);
		yi = y->col(i);
		// same for vx = v(1:nx,i), vu = v(nx+1:nx+nu,i)
		zi = z.col(i);
		xi = zi.subvec(0,nx-1);
		ui = zi.subvec(nx,nx+nu-1);
		// yi += a*(E*vx + L*vu)
		yi.gemv(E,xi,a,1.0);
		yi.gemv(L,ui,a,1.0);
	}
	y->reshape(nc*(N+1),1);
}

void MPCData::gemvG(const StaticMatrix &x, double a, double b, StaticMatrix *y){
	int nx = nx_; int nu = nu_; int N = N_;
	// TODO check sizes
	
	if(b == 0.0){
		y->fill(0.0);
	}else if(b != 1.0){
		y->operator*=(b);
	}

	y->reshape(nx,N+1);
	StaticMatrix z = x.getreshape(nx+nu,N+1);

	StaticMatrix zi,zm1;
	StaticMatrix xi,xm1,um1;
	StaticMatrix yi;
	// y(0) += -a*x(0)
	yi = y->col(0); 
	zi = z.col(0); 
	xi = zi.subvec(0,nx-1);
	yi.axpy(xi,-1.0*a);

	StaticMatrix A,B;
	for(int i=1;i<N+1;i++){
		yi = y->col(i);

		zi = z.col(i-1);
		xm1 = zi.subvec(0,nx-1);
		um1 = zi.subvec(nx,nx+nu-1);

		zi = z.col(i);
		xi = zi.subvec(0,nx-1);

		// y(i) += a*(A(i-1)*x(i-1) + B(i-1)u(i-1) - x(i))
		A = A_(i-1);
		B = B_(i-1);
		yi.gemv(A,xm1,a,1.0);
		yi.gemv(B,um1,a,1.0);
		yi.axpy(xi,-1.0*a);
	}
	y->reshape((N+1)*nx,1);
}

void MPCData::gemvGT(const StaticMatrix &x, double a, double b, StaticMatrix *y){
	int nx = nx_; int nu = nu_; int N = N_;
	// TODO check sizes
	
	if(b == 0.0){
		y->fill(0.0);
	}else if(b != 1.0){
		y->operator*=(b);
	}

	y->reshape(nx+nu,N+1);
	StaticMatrix v = x.getreshape(nx,N+1);
	
	StaticMatrix A,B;
	StaticMatrix yi,xi,ui;
	StaticMatrix vi, vp1;
	for(int i=0;i<N;i++){ // i = 0 to N-1
		yi = y->col(i);
		xi = yi.subvec(0,nx-1);
		ui = yi.subvec(nx,nx+nu-1);

		vi = v.col(i);
		vp1 = v.col(i+1);
		// xi += -v(i) + A(i)'*v(i+1)
		A = A_(i);
		xi.axpy(vi,-1.0*a);
		xi.gemv(A,vp1,1.0*a,1.0,true);

		// ui += B(i)'*v(i+1)
		B = B_(i);
		ui.gemv(B,vp1,1.0*a,1.0,true);
	}
	// i = N 
	vi = v.col(N);

	yi = y->col(N);
	xi = yi.subvec(0,nx-1);
	xi.axpy(vi,-1.0*a);

	y->reshape((nx+nu) *(N+1),1);
}

void MPCData::gemvAT(const StaticMatrix &x, double a, double b, StaticMatrix *y){
	int nx = nx_; int nu = nu_; int nc = nc_; int N = N_;
	// TODO check sizes
	
	if(b == 0.0){
		y->fill(0.0);
	}else if(b != 1.0){
		y->operator*=(b);
	}

	y->reshape(nx+nu,N+1);
	StaticMatrix v = x.getreshape(nc,N+1);
	
	StaticMatrix E,L;
	StaticMatrix yi,yx,yu;
	StaticMatrix vi;
	for(int i=0;i<N+1;i++){
		E = E_(i);
		L = L_(i);
		// get aliases for yi = y(:,i);
		yi = y->col(i);
		yx = yi.subvec(0,nx-1);
		yu = yi.subvec(nx,nx+nu-1);
		// same for vx = v(1:nx,i), vu = v(nx+1:nx+nu,i)
		vi = v.col(i);
		// yx += E'*vi, yu += L'*vi
		yx.gemv(E,vi,1.0*a,1.0,true);
		yu.gemv(L,vi,1.0*a,1.0,true);
	}

	y->reshape((N+1)*(nx+nu),1);
}

void MPCData::axpyf(double a, StaticMatrix *y){
	int nx = nx_; int nu = nu_; int N = N_;

	y->reshape(nx+nu,N+1);

	StaticMatrix yi,yx,yu;
	StaticMatrix q,r;
	for(int i = 0;i<=N;i++){
		// y(i) += a*[q(i);r(i)]
		q = q_(i);
		r = r_(i);
		
		yi = y->col(i);
		yx = yi.subvec(0,nx-1);
		yu = yi.subvec(nx,nx+nu-1);

		yx.axpy(q,a);
		yu.axpy(r,a);
	}
	y->reshape((nx+nu)*(N+1),1);
}

void MPCData::axpyh(double a, StaticMatrix *y){
	int nx = nx_; int N = N_;
	y->reshape(nx,N+1);

	// y(0) += -a*x0
	StaticMatrix yi = y->col(0);
	yi.axpy(x0_,-a);

	StaticMatrix c;
	for(int i=1;i<=N;i++){
		c = c_(i-1);
		yi = y->col(i);

		// y(i) += -a*c(i)
		yi.axpy(c,-a);
	}
	y->reshape(nx*(N+1),1);
}

void MPCData::axpyb(double a, StaticMatrix *y){
	int nc = nc_; int N = N_;
	y->reshape(nc,N+1);

	StaticMatrix yi;
	StaticMatrix d;

	for(int i=0;i<=N;i++){
		d = d_(i);
		yi = y->col(i);

		// y(i) += -a*d(i)
		yi.axpy(d,-a);
	}
	y->reshape(nc*(N+1),1);
}



}  // namespace fbstab
}  // namespace solvers
}  // namespace drake