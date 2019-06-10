#include "drake/solvers/fbstab/components/mpc_data.h"
#include "drake/solvers/fbstab/components/mpc_variable.h"
#include "drake/solvers/fbstab/components/mpc_residual.h"
#include "drake/solvers/fbstab/components/ricatti_linear_solver.h"

#include <cmath>

#include "drake/solvers/fbstab/linalg/static_matrix.h"
#include "drake/solvers/fbstab/linalg/matrix_sequence.h"
#include "drake/solvers/fbstab/test/test_helpers.h"

using namespace drake::solvers::fbstab;
using namespace std;
int main(){
	// set up the QP
	int N = 2;
	int nx = 2;
	int nu = 1;
	int nc = 6;

	// int nz = (N+1)*(nx+nu);
	// int nl = (N+1)*nx;
	// int nv = (N+1)*nc;

	double Q[] = {2,0,0,1};
	double S[] = {1,0};
	double R[] = {3};
	double q[] = {-2,0};
	double r[] = {0};

	double A[] = {1,0,1,1};
	double B[] = {0,1};
	double c[] = {0,0};
	double x0[] = {0,0};

	double E[] = {-1,0,1,0,0,0,
				  0,-1,0,1,0,0};
	double L[] = {0,0,0,0,-1,1};
	double d[] = {0,0,-2,-2,-1,-1};

	double** Qt = test::repmat(Q,2,2,N+1);
	double** Rt = test::repmat(R,1,1,N+1);
	double** St = test::repmat(S,1,2,N+1);
	double** qt = test::repmat(q,2,1,N+1);
	double** rt = test::repmat(r,1,1,N+1);

	double** At = test::repmat(A,2,2,N);
	double** Bt = test::repmat(B,2,1,N);
	double** ct = test::repmat(c,2,1,N);

	double** Et = test::repmat(E,6,2,N+1);
	double** Lt = test::repmat(L,6,1,N+1);
	double** dt = test::repmat(d,6,1,N+1);

	QPsizeMPC size = {N,nx,nu,nc};
	// data object
	MPCData data(Qt,Rt,St,qt,rt,At,Bt,ct,Et,Lt,dt,x0,size);

	MPCVariable x(size);
	MPCVariable y(size);

	x.LinkData(&data);
	y.LinkData(&data);

	x.z_.fill(1);
	x.l_.fill(2);
	x.v_.fill(4);

	y.z_.fill(2);
	y.l_.fill(1);
	y.v_.fill(3);

	x.InitConstraintMargin();
	y.InitConstraintMargin();

	double sigma = 1.0;

	RicattiLinearSolver ls(size);
	ls.LinkData(&data);
	ls.Factor(x,y,sigma);

	MPCResidual res(size);
	res.LinkData(&data);
	res.FBresidual(x,y,sigma);


	MPCVariable dx(size);
	ls.Solve(res,&dx);

	cout << ls.th_ << endl;
	cout << ls.h_<< endl;

	cout << dx.z_ << endl;
	cout << dx.l_ << endl;
	cout << dx.v_ << endl;


	test::free_repmat(Qt,N+1);
	test::free_repmat(Rt,N+1);
	test::free_repmat(St,N+1);
	test::free_repmat(qt,N+1);
	test::free_repmat(rt,N+1);

	test::free_repmat(At,N);
	test::free_repmat(Bt,N);
	test::free_repmat(ct,N);

	test::free_repmat(Et,N+1);
	test::free_repmat(Lt,N+1);
	test::free_repmat(dt,N+1);
	return 0;
}



