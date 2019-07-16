#pragma once

#include "drake/common/drake_copyable.h"

#include <vector>
#include <Eigen/Dense>

namespace drake {
namespace solvers {
namespace fbstab {

// Forward declaration of testing class to enable a friend declaration.
namespace test {
class MPCComponentUnitTests;
}  // namespace test

/**
 * This class represents data for quadratic programming problems of the
 * following type (1):
 *
 * min.  \sum_{i=0}^N 1/2 [x(i)]' * [Q(i) S(i)'] [x(i)] + [q(i)]'*[x(i)]
 *                        [u(i)]    [S(i) R(i) ] [u(i)]   [r(i)]  [u(i)]
 * s.t.  x(i+1) = A(i)*x(i) + B(i) u(i) + c(i), i = 0 ... N-1
 *       x(0) = x0,
 *       E(i)*x(i) + L(i)*u(i) + d(i) <= 0,     i = 0 ... N
 *
 * This is a specialization of the general form (2):
 *
 * min.  1/2 z'Hz + f'z
 *
 * s.t.  Gz =  h
 *       Az <= b
 *
 * This class contains storage and methods for implicitly working with the
 * compact representation (2).
 */
class MPCData {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MPCData);
  /**
   * Creates problem data and performs input validation.
   * This class assumes that the pointers to the data remain valid.
   *
   * All arguments are inputs and point to data defining a linear-quadratic
   * optimal control problem, see the class comment.
   */
  MPCData(const std::vector<Eigen::MatrixXd>* Q,
          const std::vector<Eigen::MatrixXd>* R,
          const std::vector<Eigen::MatrixXd>* S,
          const std::vector<Eigen::VectorXd>* q,
          const std::vector<Eigen::VectorXd>* r,
          const std::vector<Eigen::MatrixXd>* A,
          const std::vector<Eigen::MatrixXd>* B,
          const std::vector<Eigen::VectorXd>* c,
          const std::vector<Eigen::MatrixXd>* E,
          const std::vector<Eigen::MatrixXd>* L,
          const std::vector<Eigen::VectorXd>* d, const Eigen::VectorXd* x0);

  /**
   * Computes the operation y <- a*H*x + b*y without forming H explicitly.
   * This implements a BLAS operation, see
   * http://www.netlib.org/blas/blasqr.pdf.
   * @param[in] x Input vector, length(x) = (nx+nu)*(N+1)
   * @param[in] a Input scaling
   * @param[in] b Scaling
   * @param[both] y Output vector, length(y) = (nx+nu)*(N+1)
   */
  void gemvH(const Eigen::VectorXd& x, double a, double b,
             Eigen::VectorXd* y) const;

  /**
   * Computes y <- a*A*x + b*y without forming A explicitly.
   * This implements a BLAS operation, see
   * http://www.netlib.org/blas/blasqr.pdf.
   * @param[in] x Input vector, length(x) = (nx+nu)*(N+1)
   * @param[in] a Input scaling
   * @param[in] b Scaling
   * @param[both] y Output vector, length(y) = nc*(N+1)
   */
  void gemvA(const Eigen::VectorXd& x, double a, double b,
             Eigen::VectorXd* y) const;

  /**
   * Computes y <- a*G*x + b*y without forming G explicitly
   * This implements a BLAS operation, see
   * http://www.netlib.org/blas/blasqr.pdf.
   * @param[in] x Input vector, length(x) = (nx+nu)*(N+1)
   * @param[in] a Input scaling
   * @param[in] b Scaling
   * @param[both] y Output vector, length(y) = nx*(N+1)
   */
  void gemvG(const Eigen::VectorXd& x, double a, double b,
             Eigen::VectorXd* y) const;

  /**
   * Computes y <- a*A'*x + b*y without forming A explicitly
   * This implements a BLAS operation, see
   * http://www.netlib.org/blas/blasqr.pdf.
   * @param[in] x Input vector, length(x) = nc*(N+1)
   * @param[in] a Input scaling
   * @param[in] b Scaling
   * @param[both] y Output vector, length(y) = (nx+nu)*(N+1)
   */
  void gemvAT(const Eigen::VectorXd& x, double a, double b,
              Eigen::VectorXd* y) const;

  /**
   * Computes y <- a*G'*x + b*y without forming G explicitly
   * This implements a BLAS operation, see
   * http://www.netlib.org/blas/blasqr.pdf.
   * @param[in] x Input vector, length(x) = (nx)*(N+1)
   * @param[in] a Input scaling
   * @param[in] b Scaling
   * @param[both] y Output vector, length(y) = (nx+nu)*(N+1)
   */
  void gemvGT(const Eigen::VectorXd& x, double a, double b,
              Eigen::VectorXd* y) const;

  /**
   * Computes y <- a*f + y without forming f explicitly.
   * This implements a BLAS operation, see
   * http://www.netlib.org/blas/blasqr.pdf.
   * @param[in] a Scaling factor
   * @param[both] y Output vector, length(y) = (nx+nu)*(N+1)
   */
  void axpyf(double a, Eigen::VectorXd* y) const;

  /**
   * Computes y <- a*h + y without forming h explicitly.
   * This implements a BLAS operation, see
   * http://www.netlib.org/blas/blasqr.pdf.
   * @param[in] a Scaling factor
   * @param[both] y Output vector, length(y) = nx*(N+1)
   */
  void axpyh(double a, Eigen::VectorXd* y) const;

  /**
   * Computes y <- a*b + y without forming b explicitly.
   * This implements a BLAS operation, see
   * http://www.netlib.org/blas/blasqr.pdf.
   * @param[in] a Scaling factor
   * @param[both] y Output vector, length(y) = nc*(N+1)
   */
  void axpyb(double a, Eigen::VectorXd* y) const;

 private:
  int N_ = 0;   // horizon length
  int nx_ = 0;  // number of states
  int nu_ = 0;  // number of controls
  int nc_ = 0;  // constraints per stage
  int nz_ = 0;  // number of primal variables
  int nl_ = 0;  // number of equality duals
  int nv_ = 0;  // number of inequality duals

  const std::vector<Eigen::MatrixXd>* Q_;
  const std::vector<Eigen::MatrixXd>* R_;
  const std::vector<Eigen::MatrixXd>* S_;
  const std::vector<Eigen::VectorXd>* q_;
  const std::vector<Eigen::VectorXd>* r_;
  const std::vector<Eigen::MatrixXd>* A_;
  const std::vector<Eigen::MatrixXd>* B_;
  const std::vector<Eigen::VectorXd>* c_;
  const std::vector<Eigen::MatrixXd>* E_;
  const std::vector<Eigen::MatrixXd>* L_;
  const std::vector<Eigen::VectorXd>* d_;

  const Eigen::VectorXd* x0_;

  /**
   * Throws an exception if any of the inputs have inconsistent lengths.
   */
  void validate_length() const;
  /**
   * Throws an exception if any of the inputs have inconsistent sizes.
   * Assumes the validate_length() has already been called.
   */
  void validate_size() const;

  // friend class RicattiLinearSolver;
  friend class test::MPCComponentUnitTests;
  friend class RicattiLinearSolver;
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
