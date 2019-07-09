#pragma once

#include "drake/solvers/fbstab/components/mpc_data.h"
#include "drake/solvers/fbstab/components/mpc_residual.h"
#include "drake/solvers/fbstab/components/mpc_variable.h"
#include "drake/solvers/fbstab/linalg/matrix_sequence.h"
#include "drake/solvers/fbstab/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {

/**
 * Implements a Ricatti recursion based method for solving linear systems of
 * equations that arise when solving MPC form QPs (see mpc_data.h) using FBstab.
 * The equations have the form
 * 								 K(x,xbar,sigma)*dx =
 * r
 *
 * where dx is a primal-dual variable and r is a residual object.
 *
 * Contains workspace memory and methods for setting up and solving the linear
 * systems.
 */
class RicattiLinearSolver {
 public:
  /**
   * Allocates workspace memory.
   * @param[in] size QP dimensions
   *
   */
  RicattiLinearSolver(QPsizeMPC size);

  /**
   * Frees allocated memory.
   */
  ~RicattiLinearSolver();

  /**
   * Links to problem data needed to perform calculations.
   * @param[in] data Pointer to the data object defining the problem instance
   */
  void LinkData(MPCData* data);

  /**
   * Sets a parameter used in the algorithm
   * @param[in] alpha PFB parameter
   */
  void SetAlpha(double alpha);

  /**
   * Sets up and factors the lhs matrix K using a Ricatti recursion.
   *
   * @param[in]  x     Current inner iterate
   * @param[in]  xbar  Current outer iterate
   * @param[in]  sigma Regularization parameter
   * @return       True if the factorization succeeds, false otherwise
   *
   */
  bool Factor(const MPCVariable& x, const MPCVariable& xbar, double sigma);

  /**
   * Applies the Ricatti factorization to compute dx = inv(K)r
   * Undefined behaviour if Factor is not called first.
   *
   * @param[in]  r  rhs residual vector
   * @param[out] dx storage for the solution
   * @return    true if the solve succeeds, false otherwise
   */
  bool Solve(const MPCResidual& r, MPCVariable* dx);

  /**
   * Workspace matrices. Public to enable access for testing purposes.
   */
  // barrier augmented cost matrices
  MatrixSequence Q_;
  MatrixSequence R_;
  MatrixSequence S_;

  // Storage for the matrix portion of the recursion
  MatrixSequence P_;
  MatrixSequence SG_;
  MatrixSequence M_;
  MatrixSequence L_;
  MatrixSequence SM_;
  MatrixSequence AM_;

  // Storage for the vector portion of the recursion
  MatrixSequence h_;
  MatrixSequence th_;

  // Storage for barrier terms
  StaticMatrix gamma_;
  StaticMatrix mus_;
  StaticMatrix Gamma_;

  // Workspace matrices
  StaticMatrix Etemp_;
  StaticMatrix Linv_;

  // Workspace vectors
  StaticMatrix tx_;
  StaticMatrix tl_;
  StaticMatrix tu_;
  StaticMatrix r1_;
  StaticMatrix r2_;

 private:
  /**
   * Structure used to return two scalars
   */
  struct Point2D {
    double x;
    double y;
  };
  int N_, nx_, nu_, nc_;
  int nz_, nl_, nv_;

  MPCData* data_ = nullptr;
  double zero_tol_ = 1e-13;
  bool memory_allocated_ = false;
  double alpha_ = 0.95;

  /**
   * Computes the gradient of the penalized Fischer-Burmeister function at (a,b)
   * @param  a     input point 1
   * @param  b     input point 2
   * @return       Structure containing (d/da phi, d/db phi)
   */
  Point2D PFBgrad(double a, double b);
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
