#include "drake/solvers/dominic/fbstab_dense.h"

#include <cmath>

#include "drake/solvers/dominic/dense_components.h"
#include "drake/solvers/dominic/fbstab_algorithm.h"
#include "drake/solvers/dominic/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {


FBstabDense::FBstabDense(int n_, int q_){

	// create the component objects
	this->n = n_;
	this->q = q_;

	QPsize size = {n,q};

	// create the component objects
	DenseVariable *x1 = new DenseVariable(size);
	DenseVariable *x2 = new DenseVariable(size);
	DenseVariable *x3 = new DenseVariable(size);
	DenseVariable *x4 = new DenseVariable(size);

	DenseResidual *r1 = new DenseResidual(size);
	DenseResidual *r2 = new DenseResidual(size);

	DenseLinearSolver *linsolve = new DenseLinearSolver(size);

	// link these objects to the algorithm object
	algo = new FBstabAlgorithm(x1,x2,x3,x4,r1,r2,linsolve);
}

SolverOut FBstabDense::Solve(const QPData &qp, double *z, double *v,
	double *y, bool use_initial_guess){

	QPsize size = {n,q};

	DenseData data(qp.H,qp.f,qp.A,qp.b,size);

	DenseVariable x0(size,z,v,y);

	if(!use_initial_guess){
		x0.z.fill(0.0);
		x0.v.fill(0.0);
		x0.y.fill(0.0);
	}

	// call the solver
	return algo->Solve(&data, &x0);
}

FBstabDense::~FBstabDense(){
	// delete allocated memory
	algo->DeleteComponents();
	delete algo;
}

}  // namespace dominic
}  // namespace solvers
}  // namespace drake