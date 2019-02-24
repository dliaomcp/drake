#include "drake/solvers/dominic/fbstab_dense.h"

#include <cmath>

#include "drake/solvers/dominic/linalg/static_matrix.h"


using namespace drake::solvers::fbstab;
using namespace std;
int main(){


	// data for a simple convex qp
	double H[] = {3,1,1,1};
	double f[] = {10,5};
	double A[] = {-1,0,0,1};
	double b[] = {0,0};
	int n = 2;
	int q = 2;

	FBstabDense solver(n,q);

	QPData data;
	data.H = H;
	data.f = f;
	data.A = A;
	data.b = b;

	double z[] = {0,0};
	double v[] = {0,0};
	double y[] = {0,0};

	SolverOut status = solver.Solve(data,z,v,y);

	cout << status.eflag << endl;
	cout << status.residual << endl;
	cout << status.newton_iters << endl;
	cout << status.prox_iters << endl;

	cout << "z = ("<< z[0] << ", " << z[1] << ")\n";
	cout << "v = ("<< v[0] << ", " << v[1] << ")\n";
	cout << "y = ("<< y[0] << ", " << y[1] << ")\n";


	return 0;
}



