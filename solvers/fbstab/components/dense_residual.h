#pragma once

#include <stdexcept>

#include <Eigen/Dense>
#include "drake/solvers/fbstab/components/dense_data.h"
#include "drake/solvers/fbstab/components/dense_variable.h"

namespace drake {
namespace solvers {
namespace fbstab {

/**
 * This class computes and stores residuals for inequality constrained dense
 * QPs. See dense_data.h for a description of the QP.
 *
 * Residuals have 2 components:
 * z: Optimality residual
 * v: Complementarity residual
 */
class DenseResidual {
 public:
  /**
   * Allocates memory for computing and storing residual vectors.
   *
   * @param[in] nz Number of decision variables
   * @param[in] nv Number of inequality constraints
   * 
   * Throws an exception if any inputs aren't positive.
   */
  DenseResidual(int nz, int nv);

  /**
   * Links the residual object with the problem data needed to
   * perform calculations.
   * @param[in] data pointer to the problem data.
   */
  void LinkData(const DenseData* data) { data_ = data; };

  /**
   * Performs the operation
   * y <- -1*y (y is this object).
   */
  void Negate();

  /**
   * Computes R(x,xbar,sigma): the residual of a proximal subproblem.
   * R(x,xbar,sigma) = 0 if and only if x = P(xbar,sigma)
   * where P is the proximal operator.
   *
   * See (11) and (20) in https://arxiv.org/pdf/1901.04046.pdf
   * for a mathematical description.
   *
   * Overwrites internal storage.
   *
   * @param[in] x      Inner loop variable
   * @param[in] xbar   Outer loop variable
   * @param[in] sigma  Positive regularization strength
   */
  void InnerResidual(const DenseVariable& x, const DenseVariable& xbar,
                     double sigma);

  /**
   * Computes \pi(x): the natural residual of the QP
   * at the primal-dual point x, \pi(x) = 0 if and only if
   * x solves the QP.
   *
   * See (17) in https://arxiv.org/pdf/1901.04046.pdf
   * for a mathematical definition.
   *
   * Overwrites internal storage.
   *
   * @param[in] x Evaluation point.
   */
  void NaturalResidual(const DenseVariable& x);

  /**
   * Computes the natural residual function augmented with
   * penalty terms, it is analogous to (18) in
   * https://arxiv.org/pdf/1901.04046.pdf
   *
   * Overwrites internal storage.
   *
   * @param[in] x Evaluation point.
   */
  void PenalizedNaturalResidual(const DenseVariable& x);

  /**
   * Deep copy
   *
   * @param[in] x residual to be copied
   */
  void Copy(const DenseResidual& x);

  /**
   * Fills the storage with a i.e.,
   * r <- a*ones
   * @param[in] a
   */
  void Fill(double a);

  /**
   * Compute the Euclidean norm of the current stored residuals.
   * @return ||z|| + ||v||
   */
  double Norm() const;
  /**
   * Compute the merit function of the current stored residuals.
   * @return 0.5*(||z|| + ||v||)^2
   */
  double Merit() const;  // 2 norm squared

  /** Accessors */
  Eigen::VectorXd& z() { return z_; };
  Eigen::VectorXd& v() { return v_; };
  void SetAlpha(double alpha) { alpha_ = alpha; }
  double z_norm() const { return znorm_; }
  double v_norm() const { return vnorm_; }

  /**
   * The dense QP we consider has no equality constraints
   * so this methods returns 0.
   * It is needed by the printing routines of the FBstabAlgorithm class.
   */
  double l_norm() const { return 0; }

 private:
  const DenseData* data_ = nullptr;  // access to the data object
  int nz_ = 0;                       // number of decision variables
  int nv_ = 0;                       // number of inequality constraints
  Eigen::VectorXd z_;                // storage for the stationarity residual
  Eigen::VectorXd v_;                // storage for the complementarity residual
  double alpha_ = 0.95;
  double znorm_ = 0.0;
  double vnorm_ = 0.0;

  /*
   * Evaluate the Penalized Fischer-Burmeister function
   * (19) in https://arxiv.org/pdf/1901.04046.pdf
   */
  static double pfb(double a, double b, double alpha);

  /* Scalar max and min */
  static double max(double a, double b);
  static double min(double a, double b);

  friend class DenseLinearSolver;
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
