#pragma once

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

struct QPDataMPC {
	double **Q = nullptr;
	double **R = nullptr;
	double **S = nullptr;
	double **q = nullptr;
	double **r = nullptr;

	double **A = nullptr;
	double **B = nullptr;
	double **c = nullptr;
	double **x0 = nullptr;

	double **E = nullptr;
	double **L = nullptr;
	double **d = nullptr;
};

// Conveience type for the templated version of the algorithm
using FBstabAlgoMPC = FBstabAlgorithm<MSVariable,MSResidual,MCPCData,RicattiLinearSolver,MPCFeasibility>;

class FBstabMPC {
 public:

 	FBstabMPC(int N, int nx, int nu, int nc);
 	SolverOut Solve(const QPDataMPC &qp, double *z, double *l, double *v, double *y, bool use_initial_guess = true);

 	~FBstabMPC();

 	void UpdateOption(const char *option, double value);
 	void UpdateOption(const char *option, int value);
 	void SetDisplayLevel(FBstabAlgoDense::Display level);
 	void CheckInfeasibility(bool check);

 private:
 	int nx_,nu_,N_,nc_,nv_,nl_,nz_;
 	FBstabAlgoMPC *algo_ = nullptr;
};


}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
