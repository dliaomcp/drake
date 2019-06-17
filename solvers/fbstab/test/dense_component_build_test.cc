#define EIGEN_RUNTIME_NO_MALLOC 1

#include "drake/solvers/fbstab/components/dense_data.h"
#include "drake/solvers/fbstab/components/dense_variable.h"
#include "drake/solvers/fbstab/components/dense_residual.h"
#include "drake/solvers/fbstab/components/dense_linear_solver.h"

#include <Eigen/Dense>
#include <iostream>

using namespace drake::solvers::fbstab;
using namespace std;
using namespace Eigen;

int main(){
	Eigen::internal::set_is_malloc_allowed(true);
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

	Eigen::internal::set_is_malloc_allowed(false);
	int n = 2;
	int q = 2;

	DenseQPsize size = {n,q};
	// create data object
	DenseData data(H,f,A,b,size);

	cout << data.H_ << endl;
	cout << data.f_ << endl;
	cout << data.A_ << endl; 
	cout << data.b_ << endl;

	// variable testing *************************************
	Eigen::internal::set_is_malloc_allowed(true);
	
	DenseVariable x(size);
	DenseVariable y(size);
	x.LinkData(&data);
	y.LinkData(&data);

	Eigen::internal::set_is_malloc_allowed(false);

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
	Eigen::internal::set_is_malloc_allowed(true);
	DenseResidual r(size);
	r.LinkData(&data);
	Eigen::internal::set_is_malloc_allowed(false);

	r.NaturalResidual(x);
	r.PenalizedNaturalResidual(x);
	r.InnerResidual(x,y,1);


	// Linear solver testing
	Eigen::internal::set_is_malloc_allowed(true);
	DenseLinearSolver ls(size);
	ls.LinkData(&data);
	Eigen::internal::set_is_malloc_allowed(false);

	ls.Factor(x,y,1.0);

	ls.Solve(r,&y);


	return 0;
}



