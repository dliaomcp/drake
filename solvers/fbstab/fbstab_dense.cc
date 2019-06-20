#include "drake/solvers/fbstab/fbstab_dense.h"

#include <cmath>

#include "drake/solvers/fbstab/components/dense_data.h"
#include "drake/solvers/fbstab/components/dense_variable.h"
#include "drake/solvers/fbstab/components/dense_residual.h"
#include "drake/solvers/fbstab/components/dense_linear_solver.h"
#include "drake/solvers/fbstab/components/dense_feasibility.h"
#include "drake/solvers/fbstab/fbstab_algorithm.h"

namespace drake {
namespace solvers {
namespace fbstab {


FBstabDense::FBstabDense(int num_variables, int num_constraints){
	nz_ = num_variables;
	nv_ = num_constraints;

	DenseVariable *x1 = new DenseVariable(nz_,nv_);
	DenseVariable *x2 = new DenseVariable(nz_,nv_);
	DenseVariable *x3 = new DenseVariable(nz_,nv_);
	DenseVariable *x4 = new DenseVariable(nz_,nv_);

	DenseResidual *r1 = new DenseResidual(nz_,nv_);
	DenseResidual *r2 = new DenseResidual(nz_,nv_);

	DenseLinearSolver *linsolve = new DenseLinearSolver(nz_,nv_);
	DenseFeasibility *fcheck = new DenseFeasibility(nz_,nv_);

	// link these objects to the algorithm object
	algorithm_ = new FBstabAlgoDense(x1,x2,x3,x4,r1,r2,linsolve,fcheck);
}

FBstabDense::~FBstabDense(){
	delete algorithm_;
}

SolverOut FBstabDense::Solve(const DenseQPData &qp, const DenseQPVariable& x, bool use_initial_guess){
	DenseData data(qp.H,qp.f,qp.A,qp.b);
	DenseVariable x0(x.z,x.v,x.y);

	if(nz_ != data.num_variables() || nz_ != x0.num_variables()){
		throw std::runtime_error("In FBstabDense::Solve, Mismatch in data or initial guess variable dimension");
	}
	if(nv_ != data.num_constraints() || nv_ != x0.num_constraints()){
		throw std::runtime_error("In FBstabDense::Solve, Mismatch in data or initial guess constraint dimension");
	}
	if(!use_initial_guess){
		x0.Fill(0.0);
	}

	return algorithm_->Solve(&data, &x0);
}

void FBstabDense::UpdateOption(const char *option, int value){
	algorithm_->UpdateOption(option,value);
}
void FBstabDense::UpdateOption(const char *option, double value){
	algorithm_->UpdateOption(option,value);
}
void FBstabDense::UpdateOption(const char *option, bool value){
	algorithm_->UpdateOption(option,value);
}

void FBstabDense::SetDisplayLevel(FBstabAlgoDense::Display level){
	algorithm_->display_level() = level;
}


}  // namespace dominic
}  // namespace solvers
}  // namespace drake