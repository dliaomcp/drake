#include "drake/solvers/fbstab/dense_components.h"

#include <cmath>

#include "drake/solvers/fbstab/linalg/static_matrix.h"


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

	QPsize size = {n,q};
	// create data object
	DenseData data(H,f,A,b,size);

	cout << data.H << endl;
	cout << data.f << endl;
	cout << data.A << endl; 
	cout << data.b << endl;

	// create some variables

	DenseVariable x(size);
	x.LinkData(&data);
	x.Fill(0);
	x.InitConstraintMargin();
	cout << x;

	DenseVariable y(size);
	y.LinkData(&data);
	y.Fill(-1);
	y.ProjectDuals();
	cout << y;

	return 0;
}



