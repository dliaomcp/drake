#include "drake/solvers/fbstab/fbstab_dense.h"

#include <cmath>

#include "drake/solvers/fbstab/linalg/static_matrix.h"


using namespace drake::solvers::fbstab;
using namespace std;
int main(){


	// data for a simple convex qp
	double H[] = {3,1,1,1};
	double f[] = {10,5};
	double A[] = {-1,0,0,1};
	double b[] = {0,0};

	FBstabDense solver(2,2);

	DenseQPData data;
	data.H = H;
	data.f = f;
	data.A = A;
	data.b = b;

	double z[] = {0,0};
	double v[] = {0,0};
	double y[] = {0,0};

	solver.UpdateOption("abs_tol",1e-7);
	solver.SetDisplayLevel(FBstabAlgoDense::ITER);
	solver.Solve(data,z,v,y);

	cout << "z = ("<< z[0] << ", " << z[1] << ")\n";
	cout << "v = ("<< v[0] << ", " << v[1] << ")\n";
	cout << "y = ("<< y[0] << ", " << y[1] << ")\n";


	// unbounded below QP
	double H2[] = {1,0,0,0};
	double f2[] = {1,-1};
	double A2[] = {0,1,-1,0,0,0,0,-1};
	double b2[] = {0,3,-1,-1};

	DenseQPData data2;
	data2.H = H2;
	data2.f = f2;
	data2.A = A2;
	data2.b = b2;
	double z2[] = {0,0};
	double v2[] = {0,0,0,0};
	double y2[] = {0,0,0,0};

	FBstabDense solver2(2,4);
	solver2.SetDisplayLevel(FBstabAlgoDense::ITER);
	solver2.Solve(data2,z2,v2,y2);

	cout << "z = ("<< z2[0] << ", " << z2[1] << ")\n";

	return 0;
}



