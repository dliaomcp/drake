#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include "drake/common/drake_copyable.h"
#include "drake/solvers/fbstab/components/mpc_data.h"
#include "drake/solvers/fbstab/components/mpc_feasibility.h"
#include "drake/solvers/fbstab/components/mpc_residual.h"
#include "drake/solvers/fbstab/components/mpc_variable.h"
#include "drake/solvers/fbstab/components/ricatti_linear_solver.h"
#include "drake/solvers/fbstab/fbstab_algorithm.h"

namespace drake {
namespace solvers {
namespace fbstab {

/**
 * @file FBstabMPC implements the Proximally Stabilized Semismooth Method for
 * solving QPs for the following quadratic programming problem (1):
 *
 * min.  \sum_{i=0}^N 1/2 [x(i)]' * [Q(i) S(i)'] [x(i)] + [q(i)]'*[x(i)]
 *                        [u(i)]    [S(i) R(i) ] [u(i)]   [r(i)]  [u(i)]
 * s.t.  x(i+1) = A(i)*x(i) + B(i) u(i) + c(i), i = 0 ... N-1
 *       x(0) = x0
 *       E(i)*x(i) + L(i)*u(i) + d(i) <= 0,     i = 0 ... N
 *
 * Where
 *
 * 			 [ Q(i),S(i)']
 * 			 [ S(i),R(i) ]
 *
 * is positive semidefinite for all i \in [0,N].
 *
 * The problem is of size (N,nx,nu,nc) where
 * N > 0 is the horizon length
 * nx > 0 is the number of states
 * nu > 0 is the number of control inputs
 * nc > 0 is the number of constraints per timestep
 *
 * This is a specialization of the general form (2),
 *
 * min.  1/2 z'Hz + f'z
 *
 * s.t.  Gz =  h
 *       Az <= b
 *
 * which has dimensions nz = (nx + nu) * (N + 1), nl = nx * (N + 1),
 * and nv = nc * (N + 1).
 *
 * Aside from convexity there are no assumptions made about the problem
 * This method can detect unboundedness/infeasibility
 * and exploit arbitrary initial guesses.
 */

/**
 * Structure to hold the problem data.
 */
struct QPDataMPC {
  const std::vector<Eigen::MatrixXd>* Q = nullptr;
  const std::vector<Eigen::MatrixXd>* R = nullptr;
  const std::vector<Eigen::MatrixXd>* S = nullptr;
  const std::vector<Eigen::VectorXd>* q = nullptr;
  const std::vector<Eigen::VectorXd>* r = nullptr;
  const std::vector<Eigen::MatrixXd>* A = nullptr;
  const std::vector<Eigen::MatrixXd>* B = nullptr;
  const std::vector<Eigen::VectorXd>* c = nullptr;
  const std::vector<Eigen::MatrixXd>* E = nullptr;
  const std::vector<Eigen::MatrixXd>* L = nullptr;
  const std::vector<Eigen::VectorXd>* d = nullptr;
  const Eigen::VectorXd* x0 = nullptr;
};

/**
 * Structure to hold the initial guess and solution.
 * All vectors WILL BE OVERWRITTEN with by the solve routine.
 *
 * Fields:
 * z \in \reals^nz are the decision variables
 * l \in \reals^nl are the equality duals / costate
 * v \in \reals^nv are the inequality duals
 * y \in \reals^nv are the constraint margins, i.e., y = b - Az
 */
struct QPVariableMPC {
  Eigen::VectorXd* z = nullptr;
  Eigen::VectorXd* l = nullptr;
  Eigen::VectorXd* v = nullptr;
  Eigen::VectorXd* y = nullptr;
};

// Conveience typedef for the templated version of the algorithm.
using FBstabAlgoMPC = FBstabAlgorithm<MPCVariable, MPCResidual, MPCData,
                                      RicattiLinearSolver, MPCFeasibility>;

class FBstabMPC {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FBstabMPC);
  /**
   * Allocates workspaces needed when solving (1)
   *
   * @param[in] N Horizon length
   * @param[in] nx number of states
   * @param[in] nu number of control input
   * @param[in] nc number of constraints per timestep
   *
   * Throws a runtime_error if any inputs are negative.
   */
  FBstabMPC(int N, int nx, int nu, int nc);

  /**
   * Solves an instance of (1)
   * @param[in]   qp problem data
   * @param[both] x  initial guess, overwritten with the solution
   *
   * @param[in]   use_initial_guess if false the solver is initialized at the
   * origin
   *
   * @return      A structure containing a summary of the optimizer
   *              output. Has the following fields:
   *
   *              eflag: ExitFlag enum (see fbstab_algorithm.h)
   *                     indicating success or failure
   *              residual: Norm of the KKT residual
   *              newton_iters: Number of Newton steps taken
   *              prox_iters: Number of proximal iterations
   *
   *
   * The inputs are both structures of pointers.
   * This methods assumes that they remain valid throughout the
   * solve.
   */
  SolverOut Solve(const QPDataMPC& qp, const QPVariableMPC& x,
                  bool use_initial_guess = true);

  /**
   * Allows for setting of solver options, see fbstab_algorithm.h for a list.
   * @param option Option name
   * @param value  New value
   */
  void UpdateOption(const char* option, double value);
  void UpdateOption(const char* option, int value);
  void UpdateOption(const char* option, bool value);

  /**
   * Controls the verbosity of the algorithm.
   * @param level new display level
   *
   * Possible values are:
   * OFF:   				Silent operaton
   * FINAL: 				Prints a summary at the end
   * ITER:  				Major iterations details
   * ITER_DETAILED: Major and minor iteration details
   *
   * The default value is FINAL.
   */
  void SetDisplayLevel(FBstabAlgoMPC::Display level);

 private:
  int N_ = 0;   // horizon length
  int nx_ = 0;  // number of states
  int nu_ = 0;  // number of controls
  int nc_ = 0;  // constraints per stage
  int nz_ = 0;  // number of primal variables
  int nl_ = 0;  // number of equality duals
  int nv_ = 0;  // number of inequality duals

  std::unique_ptr<FBstabAlgoMPC> algorithm_;
  std::unique_ptr<MPCVariable> x1_;
  std::unique_ptr<MPCVariable> x2_;
  std::unique_ptr<MPCVariable> x3_;
  std::unique_ptr<MPCVariable> x4_;
  std::unique_ptr<MPCResidual> r1_;
  std::unique_ptr<MPCResidual> r2_;
  std::unique_ptr<RicattiLinearSolver> linear_solver_;
  std::unique_ptr<MPCFeasibility> feasibility_checker_;
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
