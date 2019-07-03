#pragma once

#include <Eigen/Dense>

#include "drake/solvers/fbstab/dense_components/dense_variable.h"
#include "drake/solvers/fbstab/dense_components/dense_residual.h"
#include "drake/solvers/fbstab/dense_components/dense_data.h"

namespace drake {
namespace solvers {
namespace fbstab {

/** 
 * A class for computing the search directions used by the FBstab QP Solver.
 * It solves systems of linear equations of the form
 * 
 *      [Hs   A'] dz = rz  <==>  V*dx = r
 *      [-CA  D ] dv   rv
 *      
 * using a Schur complement approach as described in (28) and (29) of 
 * https://arxiv.org/pdf/1901.04046.pdf. Note that this code doesn't have 
 * equality constraints so is a simplification of (28) and (29).
 * 
 * This class allocates its own workspace memory and splits step computation 
 * into solve and factor steps to allow for solving multiple 
 * right hand sides. 
 * 
 * Usage:
 * 
 * DenseLinearSolver solver(2,2);
 * solver.Factor(x,xbar,sigma);
 * solver.Solve(r,&dx,sigma);
 */
class DenseLinearSolver {
 public:
 	/** 
 	 * Allocates workspace memory.
 	 * @param [nz] Number of decision variables.
 	 * @param [nv] Number of inequality constraints.
 	 */
 	DenseLinearSolver(int nz, int nv);

 	/** 
 	 * Links to problem data needed to perform calculations.
 	 * @param[in] data  Pointer to problem data
 	 */
 	void LinkData(const DenseData *data){ data_ = data; };
 	
 	/** 
 	 * Factors the matrix V(x,xbar,sigma) using a Schur complement approach 
 	 * followed by a Cholesky factorization and stores the factorization
 	 * internally.
 	 * 
 	 * The matrix V is computed as described in 
 	 * Algorithm 4 of https://arxiv.org/pdf/1901.04046.pdf
 	 * 
 	 * @param[in]  x       Inner loop iterate
 	 * @param[in]  xbar    Outer loop iterate
 	 * @param[in]  sigma   Regularization strength 
 	 * @return         	   true if factorization succeeds false otherwise.
 	 */
	bool Factor(const DenseVariable &x, const DenseVariable &xbar, double sigma);

	/** 
	 * Solves the system V*x = r and stores the result in x.
	 * This method assumes that the Factor routine was run to 
	 * compute then factor the matrix V.
	 * 
	 * @param[in]   r 	The right hand side vector
	 * @param[out]  x   Overwritten with the solution
	 * @return      	true if the solve succeeds, false otherwise
	 */
	bool Solve(const DenseResidual &r, DenseVariable *x);

	/** Accessor */
	void SetAlpha(double alpha);
 private:
 	int nz_ = 0; // number of decision variables
 	int nv_ = 0; // number of inequality constraints

 	// See (19) in https://arxiv.org/pdf/1901.04046.pdf
 	double alpha_ = 0.95; 
	double zero_tolerance_ = 1e-13;

	// workspace variables
	Eigen::MatrixXd K_; 
	Eigen::VectorXd r1_;
	Eigen::VectorXd r2_;
	Eigen::VectorXd Gamma_;
	Eigen::VectorXd mus_;
	Eigen::VectorXd gamma_;
	Eigen::MatrixXd B_;

 	const DenseData *data_ = nullptr;

 	struct Point2D {double x; double y;};

 	// Computes the gradient of the PFB function 
 	// (19) in https://arxiv.org/pdf/1901.04046.pdf
 	// See section 3.3. 
 	Point2D PFBGradient(double a, double b);

 	// Solves the system A*A' x = b in place
 	// where A is lower triangular and invertible.
 	void CholeskySolve(const Eigen::MatrixXd& A, Eigen::VectorXd* b);

 	friend class DenseComponentUnitTests;
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
