#include "drake/solvers/dominic/fbstab_mpc.h"

#include "drake/solvers/dominic/linalg/static_matrix.h"
#include "drake/solvers/dominic/linalg/matrix_sequence.h"

#include "drake/solvers/dominic/components/mpc_data.h"
#include "drake/solvers/dominic/components/mpc_variable.h"
#include "drake/solvers/dominic/components/mpc_residual.h"
#include "drake/solvers/dominic/components/ricatti_linear_solver.h"
#include "drake/solvers/dominic/components/mpc_feasibility.h"



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
	MSVariable *x1 = new MSVariable(size);
	MSVariable *x2 = new MSVariable(size);
	MSVariable *x3 = new MSVariable(size);
	MSVariable *x4 = new MSVariable(size);

	MSResidual *r1 = new MSResidual(size);
	MSResidual *r2 = new MSResidual(size);

	RicattiLinearSolver *linsolve = new RicattiLinearSolver(size);
	MSVariable *fcheck = new MPCFeasibility(size);

	algo_ = new FBstabAlgoMPC(x1,x2,x3,x4,r1,r2,linsolve,fcheck);
}




}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
