#pragma once

#include "drake/solvers/fbstab/linalg/static_matrix.h"
#include "drake/solvers/fbstab/linalg/matrix_sequence.h"

#include "drake/solvers/fbstab/components/mpc_data.h"
#include "drake/solvers/fbstab/components/mpc_variable.h"
#include "drake/solvers/fbstab/components/mpc_residual.h"
#include "drake/solvers/fbstab/components/ricatti_linear_solver.h"
#include "drake/solvers/fbstab/components/mpc_feasibility.h"

#include "drake/solvers/fbstab/fbstab_algorithm.h"

namespace drake {
namespace solvers {
namespace fbstab {

/**
 * FBstabMPC implements the Proximally Stabilized Semismooth Method for solving QPs
 * for the following quadratic programming problem (1):
 *
 * min.  \sum_{i=0}^N 1/2 [x(i)]' * [Q(i) S(i)'] [x(i)] + [q(i)]'*[x(i)]
 *                        [u(i)]    [S(i) R(i) ] [u(i)]   [r(i)]  [u(i)]
 * s.t.  x(i+1) = A(i)*x(i) + B(i) u(i) + c(i), i = 0 ... N-1
 *       x(0) = x0
 *       E(i)*x(i) + L(i)*u(i) + d(i) <= 0,     i = 0 ... N
 *      
 * Where [Q(i),S(i)';S(i),R(i)] is positive semidefinite.
 *
 * Aside from convexity there are no assumptions made about the problem
 * This method can detect unboundedness/infeasibility
 * and exploit arbitrary initial guesses. 
 *
 * The problem is of size (N,nx,nu,nc) where
 * N > 0 is the horizon length
 * nx > 0 is the number of states
 * nu > 0 is the number of control inputs
 * nc > 0 is the number of constraints per timestep
 */

/**
 * Structure to hold the problem data.
 *
 * Most inputs are series of matrices stored using 
 * nested pointers, i.e., Q[i] should be a pointer to an array that stores Qi in 
 * column major format for i = 0, ..., N. (or sometimes N-1)
 *
 * Fields:
 * Q = Q0,Q1,..., QN, length(Q) = N+1, length(Q[i]) = nx*nx
 * R = R0,R1,..., RN, length(R) = N+1, length(R[i]) = nu*nu
 * S = S0,S1,..., SN, length(S) = N+1, length(S[i]) = nu*nx
 * q = q0,q1,..., qN, length(q) = N+1, length(q[i]) = nx
 * r = r0,r1,..., rN, length(r) = N+1, length(r[i]) = nu
 *
 * A = A0,A1,..., AN-1, length(A) = N, length(A[i]) = nx*nx
 * B = B0,B1,..., BN-1, length(B) = N, length(B[i]) = nx*nu
 * c = c0,c1,..., cN-1, length(c) = N, length(c[i]) = nx
 * x0, length(x0) = nx
 *
 * E = E0,E1,..., EN, length(E) = N+1, length(E[i]) = nc*nx
 * L = L0,L1,..., LN, length(L) = N+1, length(L[i]) = nc*nu
 * d = d0,d1,..., dN, length(d) = N+1, sizeof(d[i]) = nc
 *
 * 
 * E.g., if Q0 =  [1 3],  Q1 = [5 7]
 * 				  [2 4]        [6 8]
 *
 * Then Q[0] should point to {1,2,3,4} and Q[1] should point to {5,6,7,8}
 *
 */
struct QPDataMPC {
	double **Q = nullptr;
	double **R = nullptr;
	double **S = nullptr;
	double **q = nullptr;
	double **r = nullptr;

	double **A = nullptr;
	double **B = nullptr;
	double **c = nullptr;
	double *x0 = nullptr;

	double **E = nullptr;
	double **L = nullptr;
	double **d = nullptr;
};

// Conveience type for the templated version of the algorithm
using FBstabAlgoMPC = FBstabAlgorithm<MPCVariable,MPCResidual,MPCData,RicattiLinearSolver,MPCFeasibility>;
class FBstabMPC {
 public:

 	/**
 	 * Allocates workspaces needed when solving (1)
 	 *
 	 * @param[in] N Horizon length
 	 * @param[in] nx number of states
 	 * @param[in] nu number of control input
 	 * @param[in] nc number of constraints per timestep
 	 */
 	FBstabMPC(int N, int nx, int nu, int nc);

 	/**
 	 * Free allocated workspace memory
 	 */
 	~FBstabMPC();

 	/**
 	 * Solves an instance of (1)
 	 * @param[in]  qp                Structure containing the problem data, see
 	 * 								 structure definition for input format
 	 * 								 
 	 * @param[both]  z               Initial guess for the primal variable
 	 *                           	 z = (x0,u0,...,uN,xN). Overwritten with the
 	 *                               solution
 	 *                               
 	 * @param[both]  l               Initial guess for the co-states. Overwritten 
 	 * 								 with the solution
 	 * 								 
 	 * @param[both]  v               Initial guess for the inequality duals. 
 	 * 								 Used to encode constraint activity status,
 	 *                               vi = 0 implies the ith constraint is inactive
 	 * 								 Overwritten with the solution.
 	 * 								 
 	 * @param[out]  y                Vector of constraint margins, y >= 0 indicates
 	 *                               constraint satisfaction
 	 *                               
 	 * @param[in]  use_initial_guess If false the solver is 
 	 * 								 initialized at the origin
 	 * 								 
 	 * @return                   A structure containing a summary of the optimizer
 	 *                           output. Has the following fields:
 	 *                           
 	 *                           eflag: ExitFlag enum (see fbstab_algorithm.h)
 	 *                           indicating success or failure
 	 *                           residual: Norm of the KKT residual
 	 *                           newton_iters: Number of Newton steps taken
 	 *                           prox_iters: Number of proximal iterations
 	 *                           
 	 */
 	SolverOut Solve(const QPDataMPC &qp, 
 		double *z, double *l, double *v, double *y, bool use_initial_guess = true);

 	/**
 	 * Allows for setting of solver options. See fbstab_algorithm.h for 
 	 * a list of adjustable options
 	 * @param option Option name
 	 * @param value  New value
 	 */
 	void UpdateOption(const char *option, double value);
 	void UpdateOption(const char *option, int value);
 	void UpdateOption(const char *option, bool value);

 	/**
 	 * Controls the verbosity of the algorithm.
 	 * @param level new display level
 	 *
 	 * Possible values are
 	 * OFF: Silent operaton
 	 * FINAL: Prints a summary at the end
 	 * ITER: Major iterations details
 	 * ITER_DETAILED: Major and minor iteration details
 	 *
 	 * The default value is FINAL
 	 */
 	void SetDisplayLevel(FBstabAlgoMPC::Display level);

 private:
 	int nx_,nu_,N_,nc_,nv_,nl_,nz_;
 	FBstabAlgoMPC *algo_ = nullptr;
};


}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
