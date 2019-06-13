#include "drake/solvers/fbstab/fbstab_mpc.h"

#include "drake/solvers/fbstab/linalg/static_matrix.h"
#include "drake/solvers/fbstab/linalg/matrix_sequence.h"

#include "drake/solvers/fbstab/components/mpc_data.h"
#include "drake/solvers/fbstab/components/mpc_variable.h"
#include "drake/solvers/fbstab/components/mpc_residual.h"
#include "drake/solvers/fbstab/components/ricatti_linear_solver.h"
#include "drake/solvers/fbstab/components/mpc_feasibility.h"

namespace drake {
namespace solvers {
namespace fbstab {

FBstabMPC::FBstabMPC(int N, int nx, int nu, int nc){
	if(N < 1 || nx < 1 || nu < 1 || nc < 1)
		throw std::runtime_error("Invalid size for MPC problem");

	nx_ = nx;
	nu_ = nu;
	nc_ = nc;
	N_ = N;
	nz_ = (nx+nu)*(N+1);
	nl_ = nx*(N+1);
	nv_ = nc*(N+1);

	QPsizeMPC size;
	size.N = N;
	size.nx = nx;
	size.nu = nu;
	size.nc = nc;

	// create the components
	MPCVariable *x1 = new MPCVariable(size);
	MPCVariable *x2 = new MPCVariable(size);
	MPCVariable *x3 = new MPCVariable(size);
	MPCVariable *x4 = new MPCVariable(size);

	MPCResidual *r1 = new MPCResidual(size);
	MPCResidual *r2 = new MPCResidual(size);

	RicattiLinearSolver *linsolve = new RicattiLinearSolver(size);
	MPCFeasibility *fcheck = new MPCFeasibility(size);

	algo_ = new FBstabAlgoMPC(x1,x2,x3,x4,r1,r2,linsolve,fcheck);
}

SolverOut FBstabMPC::Solve(const QPDataMPC &qp, double *z, double *l, double *v, double *y, bool use_initial_guess){
	QPsizeMPC size;
	size.nx = nx_;
	size.nu = nu_;
	size.nc = nc_;
	size.N = N_;

	MPCData data(qp.Q,qp.R,qp.S,qp.q,qp.r,qp.A,qp.B,qp.c,qp.E,qp.L,qp.d,qp.x0,size);
	MPCVariable x0(size,z,l,v,y);

	if(!use_initial_guess){
		x0.z().fill(0.0);
		x0.l().fill(0.0);
		x0.v().fill(0.0);
		x0.y().fill(0.0);
	}

	return algo_->Solve(&data,&x0);
}

void FBstabMPC::UpdateOption(const char *option, double value){
	algo_->UpdateOption(option,value);
}
void FBstabMPC::UpdateOption(const char *option, int value){
	algo_->UpdateOption(option,value);
}
void FBstabMPC::UpdateOption(const char *option, bool value){
	algo_->UpdateOption(option,value);
}
void FBstabMPC::SetDisplayLevel(FBstabAlgoMPC::Display level){
	algo_->display_level() = level;
}

FBstabMPC::~FBstabMPC(){
	algo_->DeleteComponents();
	delete algo_;
}


}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
