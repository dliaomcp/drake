#pragma once

#include "drake/solvers/fbstab/components/mpc_data.h"
#include "drake/solvers/fbstab/components/mpc_variable.h"
#include "drake/solvers/fbstab/linalg/matrix_sequence.h"
#include "drake/solvers/fbstab/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {

/**
 * A class that computes and stores residuals for MPC QPs. See data.h
 * for the mathematical description.
 * Residuals have 3 fields:
 * z: Optimality residual
 * l: Equality residual
 * v: Inequality/complimentarity constraint residual
 */
class MPCResidual {
 public:
  /**
   * Allocates memory for computing QP residuals
   *
   * size has the following fields
   * N:  horizon length
   * nx: number of states
   * nu: number of control input
   * nc: number of constraints per stage
   */
  MPCResidual(QPsizeMPC size);

  /**
   * Frees allocated memory.
   */
  ~MPCResidual();

  /**
   * Links the residual object to problem data needed to perform calculations
   * Calculations cannot be performed until a data object is provided
   * @param[in] data Pointer to the problem data
   */
  void LinkData(MPCData* data);

  /**
   * Sets the value of alpha used in residual computations
   * @param[in] alpha value to be set
   */
  void SetAlpha(double alpha);

  /**
   * Filles memory with vectors of a
   * @param[in] a value to fill
   */
  void Fill(double a);

  /**
   * Deep copy of x into this.
   * @param[in] x residual to be copied
   */
  void Copy(const MPCResidual& x);

  /**
   * Sets y <- -1*y where y is the residual
   */
  void Negate();

  /**
   * Euclidean norm of the residual ||F||_2
   * @return norm of the residual
   */
  double Norm() const;

  /**
   * Computes the merit function 0.5 ||F||^2_2
   * @return Merit function value
   */
  double Merit() const;

  /**
   * Computes the one norm of the residual ||F||_1
   * @return one norm
   */
  double AbsSum() const;

  /**
   * Computes the proximal subproblem residual at the point x
   * relative to the reference point xbar with regularization strength sigma.
   * Corresponds to (20) in https://arxiv.org/pdf/1901.04046.pdf
   * Overwrites internal storage
   * @param[in] x     inner iterate for evaluating the residual
   * @param[in] xbar  outer iterate for evaluating the residual
   * @param[in] sigma regularization strength
   */
  void InnerResidual(const MPCVariable& x, const MPCVariable& xbar,
                     double sigma);

  /**
   * Computes the natural residual of the KKT conditions at x.
   * Overwrites internal storage.
   * Corresponds to (23) in https://arxiv.org/pdf/1901.04046.pdf
   * @param[in] x primal-dual point to evaluate the KKT conditions
   */
  void NaturalResidual(const MPCVariable& x);

  /**
   * Computes the penalized natural residual of the KKT conditions at x
   * Overwrites internal storage
   * @param[in] x primal-dual point to evaluate the KKT conditions
   */
  void PenalizedNaturalResidual(const MPCVariable& x);

  StaticMatrix z() { return z_; }
  StaticMatrix l() { return l_; }
  StaticMatrix v() { return v_; }

  double z_norm() const { return znorm_; }
  double l_norm() const { return lnorm_; }
  double v_norm() const { return vnorm_; }

 private:
  StaticMatrix z_;
  StaticMatrix l_;
  StaticMatrix v_;

  int N_ = 0;   // horizon length
  int nx_ = 0;  // number of states
  int nu_ = 0;  // number of controls
  int nc_ = 0;  // constraints per stage
  int nz_ = 0;  // number of primal variables
  int nl_ = 0;  // number of equality duals
  int nv_ = 0;  // number of inequality duals

  double alpha_ = 0.95;
  MPCData* data_ = nullptr;

  double znorm_ = 0.0;
  double vnorm_ = 0.0;
  double lnorm_ = 0.0;

  /**
   * Computes the penalized Fischer-Burmeister function pfb(a,b)
   * Equation (19) of https://arxiv.org/pdf/1901.04046.pdf
   * @param[in]  a     input 1
   * @param[in]  b     input 2
   * @param[in]  alpha weighting parameter
   * @return       computed value
   */
  static double pfb(double a, double b, double alpha);
  static double max(double a, double b);
  static double min(double a, double b);

  bool memory_allocated_ = false;
  friend class RicattiLinearSolver;
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake