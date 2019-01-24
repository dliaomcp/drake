#include "drake/solvers/dominic/dense_components.h"

#include <cmath>

#include "drake/solvers/dominic/linalg/static_matrix.h"




using namespace drake::solvers::dominic;
using namespace std;
int main(){


	// data for a simple convex qp
	double H[] = {3,1,1,1};
	double f[] = {1,6};
	double A[] = {-1,0,0,1};
	double b[] = {0,0};
	int n = 2;
	int q = 2;

	QPsize size = {n,q};
	// create data object
	DenseData data(H,f,A,b,size);

	cout << data.H << endl;
	cout << data.f << endl;
	cout << data.A << endl; 
	cout << data.b << endl;

	// create some variables

	DenseVariable x(size);
	


	return 0;
}



