#include "drake/solvers/fbstab/components/dense_data.h"
#include "drake/solvers/fbstab/components/dense_variable.h"
#include "drake/solvers/fbstab/components/dense_residual.h"
#include "drake/solvers/fbstab/components/dense_linear_solver.h"
#include "drake/solvers/fbstab/test/dense_component_unit_tests.h"

#include <Eigen/Dense>
#include <iostream>

using namespace drake::solvers::fbstab;
using namespace std;
using namespace Eigen;

int main(){
	MatrixXd H(2,2);
	MatrixXd A(2,2);
	VectorXd f(2);
	VectorXd b(2);

	H << 3,1,
		 1,1;
	A << -1,0,
		  0,1;
	f << 1,6;
	b << 0,0;

	int n = 2;
	int q = 2;

	DenseQPsize size = {n,q};
	// create data object
	DenseData data(H,f,A,b);

	cout << data.H_ << endl;
	cout << data.f_ << endl;
	cout << data.A_ << endl; 
	cout << data.b_ << endl;

	// variable testing *************************************
	
	DenseVariable x(size);
	DenseVariable y(size);
	x.LinkData(&data);
	y.LinkData(&data);


	x.Fill(1.0);
	y.Fill(-1);

	x.InitializeConstraintMargin();

	y.ProjectDuals();
	cout << y << endl;

	// x <- a*y + x
	x.axpy(y,2.0);

	cout << x << endl;


	VectorXd& xz = x.z();

	xz(0) = 5;

	cout << x << endl;

	y.Copy(x);

	// residual testing *************************************
	DenseResidual r(size);
	r.LinkData(&data);

	r.NaturalResidual(x);
	r.PenalizedNaturalResidual(x);
	r.InnerResidual(x,y,1);


	// Linear solver testing
	DenseLinearSolver ls(size);
	ls.LinkData(&data);

	ls.Factor(x,y,1.0);

	ls.Solve(r,&y);



	return 0;
}



