#include "drake/solvers/fbstab/fbstab_dense.h"

#include <Eigen/Dense>
#include <memory>

#include "drake/solvers/fbstab/dense_components/dense_data.h"
#include "drake/solvers/fbstab/dense_components/dense_variable.h"
#include "drake/solvers/fbstab/dense_components/dense_residual.h"
#include "drake/solvers/fbstab/dense_components/dense_linear_solver.h"
#include "drake/solvers/fbstab/dense_components/dense_feasibility.h"
#include "drake/solvers/fbstab/algorithm/fbstab_algorithm.h"

namespace drake {
namespace solvers {
namespace fbstab {


FBstabDense::FBstabDense(int num_variables, int num_constraints){
	nz_ = num_variables;
	nv_ = num_constraints;

	x1_ = std::make_unique<DenseVariable>(nz_, nv_);
	x2_ = std::make_unique<DenseVariable>(nz_, nv_);
	x3_ = std::make_unique<DenseVariable>(nz_, nv_);
	x4_ = std::make_unique<DenseVariable>(nz_, nv_);

	r1_ = std::make_unique<DenseResidual>(nz_, nv_);
	r2_ = std::make_unique<DenseResidual>(nz_, nv_);

	linear_solver_ = std::make_unique<DenseLinearSolver>(nz_, nv_);
	feasibility_checker_ = std::make_unique<DenseFeasibility>(nz_, nv_);

	algorithm_ = std::make_unique<FBstabAlgoDense>(x1_.get(),x2_.get(),x3_.get(),x4_.get(),r1_.get(),r2_.get(),linear_solver_.get(),feasibility_checker_.get());
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


}  // namespace fbstab
}  // namespace solvers
}  // namespace drake