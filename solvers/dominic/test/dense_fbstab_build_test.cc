#include "drake/solvers/dominic/fbstab_dense.h"

#include <cmath>

#include "drake/solvers/dominic/linalg/static_matrix.h"


using namespace drake::solvers::fbstab;
using namespace std;
int main(){


	// data for a simple convex qp
	double H[] = {3,1,1,1};
	double f[] = {1,6};
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

	return 0;
}



