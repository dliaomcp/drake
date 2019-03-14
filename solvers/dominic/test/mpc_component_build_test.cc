#include "drake/solvers/dominic/components/mpc_components.h"

#include <cmath>

#include "drake/solvers/dominic/linalg/static_matrix.h"
#include "drake/solvers/dominic/linalg/matrix_sequence.h"
#include "drake/solvers/dominic/test/test_helpers.h"

using namespace drake::solvers::fbstab;
using namespace std;
int main(){
	// set up the QP
	int N = 2;
	int nx = 2;
	int nu = 1;
	int nc = 6;

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

	int nz = (N+1)*(nx+nu);
	int nl = (N+1)*nx;
	int nv = (N+1)*nc;

	// create inputs
	double* zm = new double[nz];
	double* lm = new double[nl];
	double* vm = new double[nv];
	StaticMatrix z(zm,nz);
	StaticMatrix l(lm,nl);
	StaticMatrix v(vm,nv);
	for(int i = 0;i<z.len();i++) 
		z(i) = i+1;
	for(int i = 0;i<l.len();i++) 
		l(i) = i+1;
	for(int i = 0;i<v.len();i++) 
		v(i) = i+1;

	// create outputs
	double* y1m = new double[nz];
	double* y2m = new double[nl];
	double* y3m = new double[nv];
	double* y4m = new double[nz];
	double* y5m = new double[nz];
	StaticMatrix y1(y1m,nz); y1.fill(0);
	StaticMatrix y2(y2m,nl); y2.fill(0);
	StaticMatrix y3(y3m,nv); y3.fill(0);
	StaticMatrix y4(y4m,nz); y4.fill(0);
	StaticMatrix y5(y5m,nz); y5.fill(0);

	data.gemvH(z,-2.0,0.0,&y1);
	cout << y1 << endl;

	data.gemvG(z,-2.0,0.0,&y2);
	cout << y2 << endl;

	data.gemvA(z,-2.0,0.0,&y3);
	cout << y3 << endl;

	data.gemvGT(l,-2.0,0.0,&y4);
	cout << y4 << endl;

	data.gemvAT(v,-2.0,0.0,&y5);
	cout << y5 << endl;


	delete[] zm;
	delete[] lm;
	delete[] vm;
	delete[] y1m;
	delete[] y2m; 
	delete[] y3m;
	delete[] y4m;
	delete[] y5m;

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



