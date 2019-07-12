#pragma once

#include <Eigen/Dense>
#include "drake/solvers/fbstab/components/dense_data.h"
#include "drake/solvers/fbstab/components/dense_variable.h"

namespace drake {
namespace solvers {
namespace fbstab {

/**
 * A class for detecting infeasibility in quadratic programs, see
 * dense_data.h for a description.
 *
 * It contains methods for determining if a primal-dual variable
 * is a certificate of either unboundedness (dual infeasibility)
 * or primal infeasibility. It implements
 * Algorithm 3 of https://arxiv.org/pdf/1901.04046.pdf
 */
class DenseFeasibility {
 public:
  /**
   * Allocates workspace memory.
   * @param[in] nz number of decision variables
   * @param[in] nv number of inequality constraints
   */
  DenseFeasibility(int nz, int nv);

  /**
   * Links to problem data needed to perform calculations.
   * @param[in] data  Pointer to problem data
   */
  void LinkData(const DenseData* data) { data_ = data; };

  /**
   * Checks if the primal-dual variable x
   * is a certificate of infeasibility and
   * stores the results internally.
   *
   * It uses the results from Proposition 4 of
   * https://arxiv.org/pdf/1901.04046.pdf
   *
   * @param[in] x   Variable to check
   * @param[in] tol Numerical tolerance
   */
  void ComputeFeasibility(const DenseVariable& x, double tol);

  /**
   * Returns the results of ComputeFeasibility
   * @return false if the last point checked certifies that
   *               QP is unbounded below, true otherwise
   */
  bool IsDualFeasible();

  /**
   * Returns the results of ComputeFeasibility
   * @return false if the last point checked certifies that
   *               QP is infeasible, true otherwise
   */
  bool IsPrimalFeasible();

 private:
  int nz_ = 0;  // number of decision variables
  int nv_ = 0;  // number of inequality constraints

  // workspace
  Eigen::VectorXd z1_;
  Eigen::VectorXd v1_;

  bool primal_feasible_ = true;
  bool dual_feasible_ = true;
  const DenseData* data_ = nullptr;
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
