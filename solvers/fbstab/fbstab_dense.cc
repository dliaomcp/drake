#include "drake/solvers/fbstab/fbstab_dense.h"

#include <cmath>

#include "drake/solvers/fbstab/components/dense_data.h"
#include "drake/solvers/fbstab/components/dense_variable.h"
#include "drake/solvers/fbstab/components/dense_residual.h"
#include "drake/solvers/fbstab/components/dense_linear_solver.h"
#include "drake/solvers/fbstab/components/dense_feasibility.h"
#include "drake/solvers/fbstab/fbstab_algorithm.h"
#include "drake/solvers/fbstab/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {


FBstabDense::FBstabDense(int n, int q){
	n_ = n;
	q_ = q;

	DenseQPsize size = {n,q};

	// create the component objects
	DenseVariable *x1 = new DenseVariable(size);
	DenseVariable *x2 = new DenseVariable(size);
	DenseVariable *x3 = new DenseVariable(size);
	DenseVariable *x4 = new DenseVariable(size);

	DenseResidual *r1 = new DenseResidual(size);
	DenseResidual *r2 = new DenseResidual(size);

	DenseLinearSolver *linsolve = new DenseLinearSolver(size);
	DenseFeasibility *fcheck = new DenseFeasibility(size);

	// link these objects to the algorithm object
	algo = new FBstabAlgoDense(x1,x2,x3,x4,r1,r2,linsolve,fcheck);
}

SolverOut FBstabDense::Solve(const DenseQPData &qp, double *z, double *v,
	double *y, bool use_initial_guess){

	DenseQPsize size = {n_,q_};
	DenseData data(qp.H,qp.f,qp.A,qp.b,size);
	DenseVariable x0(size,z,v,y);

	if(!use_initial_guess){
		x0.z().fill(0.0);
		x0.v().fill(0.0);
		x0.y().fill(0.0);
	}

	// call the solver
	return algo->Solve(&data, &x0);
}

void FBstabDense::UpdateOption(const char *option, int value){
	algo->UpdateOption(option,value);
}
void FBstabDense::UpdateOption(const char *option, double value){
	algo->UpdateOption(option,value);
}
void FBstabDense::UpdateOption(const char *option, bool value){
	algo->UpdateOption(option,value);
}

void FBstabDense::SetDisplayLevel(FBstabAlgoDense::Display level){
	algo->display_level() = level;
}

FBstabDense::~FBstabDense(){
	// delete allocated memory
	algo->DeleteComponents();
	delete algo;
}

}  // namespace dominic
}  // namespace solvers
}  // namespace drake