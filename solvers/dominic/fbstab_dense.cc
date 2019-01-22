#include "drake/solvers/dominic/fbstab_dense.h"

#include <cmath>

#include "drake/solvers/dominic/dense_components.h"
#include "drake/solvers/dominic/fbstab_algorithm.h"
#include "drake/solvers/dominic/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {


FBstabDense::FBstabDense(int n, int q){

	// create the component objects
	this->n = n;
	this->q = q;

	QPsize size = {n,q};

	DenseVariable x1(size);
	DenseVariable x2(size);
	DenseVariable x3(size);
	DenseVariable x4(size);

	DenseResidual r1(size);
	DenseResidual r2(size);

	DenseLinearSolver linsolve(size);

	// create the algorithm object
	FBstabAlgorithm algo(&x1,&x2,&x3,&x4,&r1,&r2,&linsolve);

}

SolverOut FBstabDense::Solve(const QPdata &qp, double *z, double *v,
	double *y, bool use_initial_guess){

	QPsize size = {n,q};
	// create the data object
	DenseData data(qp.H,qp.f,qp.A,qp.b);
	// create the initial condition
	DenseVariable x0(size,z,v,y);

	if(!use_initial_guess){
		x0.z.fill(0.0);
		x0.v.fill(0.0);
		x0.y.fill(0.0);
	}

	// call the solver
	return algo.Solve(data, &x0);
}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake