#pragma once

#include "drake/solvers/dominic/linalg/static_matrix.h"
#include "drake/solvers/dominic/linalg/matrix_sequence.h"
#include "drake/solvers/dominic/components/mpc_data.h"
#include "drake/solvers/dominic/components/mpc_variable.h"

namespace drake {
namespace solvers {
namespace fbstab {

class MPCResidual{
 public:

 	/**
 	 * Allocates memory for computing QP residuals for an MPC
 	 * problem of a given size
 	 * 
 	 * size has the following fields
 	 * N:  horizon length
 	 * nx: number of states
 	 * nu: number of control input
 	 * nc: number of constraints per stage
 	 */
 	MPCResidual(QPsizeMPC size);

 	/**
 	 * Frees allocated memory
 	 */
 	~MPCResidual();

 	/**
 	 * Links the residual object to problem data needed to perform calculations
 	 * Calculations cannot be performed until a data object is provided
 	 * @param[in] data Pointer to the problem data
 	 */
 	void LinkData(MPCData *data);

 	/**
 	 * Sets the value of alpha used in residual computations
 	 * @param alpha value to be set
 	 */
 	void SetAlpha(double alpha);

 	/**
 	 * Filles memory with vectors of a
 	 * @param a value to fill
 	 */
 	void Fill(double a);

 	/**
 	 * Deep copy of x into the the object.
 	 * Overwrites internal storage.
 	 * @param[in] x residual to be copied
 	 */
 	void Copy(const MPCResidual &x);

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
 	 * Overwrites internal storage
 	 * @param[in] x     inner iterate for evaluating the residual
 	 * @param[in] xbar  outer iterate for evaluating the residual
 	 * @param[in] sigma regularization strength
 	 */
 	void FBresidual(const MPCVariable &x, const MPCVariable &xbar, double sigma);

 	/**
 	 * Computes the natural residual of the KKT conditions at x.
 	 * Overwrites internal storage.
 	 * @param[in] x primal-dual point to evaluate the KKT conditions
 	 */
 	void NaturalResidual(const MPCVariable &x);

 	/**
 	 * Computes the penalized natural residual of the KKT conditions at x
 	 * Overwrites internal storage
 	 * @param[in] x primal-dual point to evaluate the KKT conditions
 	 */
 	void PenalizedNaturalResidual(const MPCVariable &x);
 	
 	StaticMatrix z_; // primal/optimality residual
 	StaticMatrix l_; // equality constraint residual
 	StaticMatrix v_; // inequality constraint residual

 	double z_norm = 0.0;
 	double v_norm = 0.0;
 	double l_norm = 0.0;

 private:
 	// problem sizes
 	int N_, nx_, nu_, nc_;
 	int nz_, nl_, nv_;

 	double alpha_ = 0.95;
 	MPCData *data_ = nullptr;

 	/**
 	 * Computes the penalized Fischer-Burmeister function phi(a,b)
 	 * @param[in]  a     input 1
 	 * @param[in]  b     input 2
 	 * @param[in]  alpha weighting parameter
 	 * @return       computed value
 	 */
 	static double pfb(double a, double b, double alpha);
 	static double max(double a, double b);
 	static double min(double a, double b);

 	bool memory_allocated_ = false;
};


}  // namespace fbstab
}  // namespace solvers
}  // namespace drake