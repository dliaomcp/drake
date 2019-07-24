#pragma once

#include <vector>

#include <Eigen/Dense>

#include "drake/common/drake_copyable.h"
#include "drake/solvers/fbstab/components/mpc_data.h"
#include "drake/solvers/fbstab/components/mpc_residual.h"
#include "drake/solvers/fbstab/components/mpc_variable.h"

namespace drake {
namespace solvers {
namespace fbstab {

// Forward declaration of testing class to enable a friend declaration.
namespace test {
class MPCComponentUnitTests;
}  // namespace test

/**
 * Implements a Ricatti recursion based method for solving linear systems of
 * equations that arise when solving MPC form QPs (see mpc_data.h) using FBstab.
 * The equations are of the form
 *
 * [Hs  G' A'][dz] = [rz]
 * [-G  sI 0 ][dl] = [rl]
 * [-CA 0  D ][dv] = [rv]
 *
 * where s = sigma, C = diag(gamma), D = diag(mu + sigma*gamma).
 * The vectors gamma and mu are defined in (24) of
 * https://arxiv.org/pdf/1901.04046.pdf.
 *
 * In compact form: V(x,xbar,sigma)*dx = r.
 *
 * This classes uses a Ricatti recursion like the one in
 *
 * Rao, Christopher V., Stephen J. Wright, and James B. Rawlings.
 * "Application of interior-point methods to model predictive control."
 * Journal of optimization theory and applications 99.3 (1998): 723-757.
 *
 * to perform the factorization efficiently. This class contains workspace
 * memory and methods for setting up and solving the linear systems.
 */
class RicattiLinearSolver {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(RicattiLinearSolver);
  /**
   * Allocates workspace memory.
   *
   * @param[in] N  horizon length
   * @param[in] nx number of states
   * @param[in] nu number of control input
   * @param[in] nc number of constraints per stage
   */
  RicattiLinearSolver(int N, int nx, int nu, int nc);

  /**
   * Sets a parameter used in the algorithm, see (19)
   * in https://arxiv.org/pdf/1901.04046.pdf.
   * @param[in] alpha
   */
  void SetAlpha(double alpha) { alpha_ = alpha; };

  /**
   * Evaluates V then factors it using a Ricatti recursion.
   *
   * @param[in]  x     Current inner iterate
   * @param[in]  xbar  Current outer iterate
   * @param[in]  sigma regularization parameter
   * @return     true if the factorization succeeds, false otherwise
   *
   * Throws a runtime_error if problem data hasn't been linked,
   * x and xbar aren't matched in size, or sigma isn't positive.
   */
  bool Factor(const MPCVariable& x, const MPCVariable& xbar, double sigma);

  /**
   * Applies the Ricatti factorization to compute dx = inv(V)*r,
   * Factor MUST be called first.
   *
   * @param[in]  r    rhs residual
   * @param[out] dx   storage for the solution
   * @return     true if the solve succeeds, false otherwise
   *
   * Throws a runtime_error if problem data hasn't been linked or
   * if the sizes of dx and r don't match.
   */
  bool Solve(const MPCResidual& r, MPCVariable* dx) const;

 private:
  // Workspace matrices.
  std::vector<Eigen::MatrixXd> Q_;
  std::vector<Eigen::MatrixXd> R_;
  std::vector<Eigen::MatrixXd> S_;

  // Storage for the matrix portion of the recursion.
  std::vector<Eigen::MatrixXd> P_;
  std::vector<Eigen::MatrixXd> SG_;
  std::vector<Eigen::MatrixXd> M_;
  std::vector<Eigen::MatrixXd> L_;
  std::vector<Eigen::MatrixXd> SM_;
  std::vector<Eigen::MatrixXd> AM_;

  // Storage for the vector portion of the recursion.
  mutable std::vector<Eigen::VectorXd> h_;
  mutable std::vector<Eigen::VectorXd> th_;

  // Workspace.
  Eigen::VectorXd gamma_;
  Eigen::VectorXd mus_;
  Eigen::MatrixXd Gamma_;

  // Workspace matrices.
  Eigen::MatrixXd Etemp_;
  Eigen::MatrixXd Ltemp_;
  Eigen::MatrixXd Linv_;

  // Workspace vectors.
  mutable Eigen::VectorXd tx_;
  mutable Eigen::VectorXd tl_;
  mutable Eigen::VectorXd tu_;
  mutable Eigen::VectorXd r1_;
  mutable Eigen::VectorXd r2_;
  mutable Eigen::VectorXd r3_;

  int N_ = 0;   // horizon length
  int nx_ = 0;  // number of states
  int nu_ = 0;  // number of controls
  int nc_ = 0;  // constraints per stage
  int nz_ = 0;  // number of primal variables
  int nl_ = 0;  // number of equality duals
  int nv_ = 0;  // number of inequality duals

  const MPCData* data_ = nullptr;
  double zero_tol_ = 1e-13;
  double alpha_ = 0.95;

  /**
   * Computes the gradient of phi at (a,b),
   * see (19) in https://arxiv.org/pdf/1901.04046.pdf.
   * @param[in]  a
   * @param[in]  b
   * @return     d/da phi(a,b) and d/db phi(a,b)
   */
  Eigen::Vector2d PFBgrad(double a, double b);

  friend class test::MPCComponentUnitTests;
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
