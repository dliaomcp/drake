#pragma once

#include <Eigen/Dense>

#include "drake/solvers/fbstab/dense_components/dense_data.h"

namespace drake {
namespace solvers {
namespace fbstab {

/** 
 * Implements primal-dual variables for inequality constrained QPs. 
 * See dense_data.h for a mathematical description.
 * This class stores variables and defines methods implementing useful operations.
 * 
 * Primal-dual variables have 3 components:
 * z: Decision variables
 * v: Inequality duals
 * y: Inequality margins
 * 
 * where
 * length(z) = nz
 * length(v) = nv
 * length(y) = nv
 */
class DenseVariable{
 public:
 	/** 
 	 * Allocates memory for a primal-dual variables.
 	 * 
 	 * @param[in] nz Number of decision variables
 	 * @param[in] nv Number of inequality constraints
 	 */
	DenseVariable(int nz, int nv);

	/** 
	 * Creates a primal-dual variable using preallocated memory.
	 * @param[in] z    A vector to store the decision variables
	 * @param[in] v    A vector to store the dual variables
	 * @param[in] y    A vector to store the inequality margin
	 * 
	 * Checks to ensure that v.size() == y.size().
	 */
	DenseVariable(Eigen::VectorXd* z, Eigen::VectorXd* v, Eigen::VectorXd* y);

	/** 
	 * Frees memory if it was allocated.
	 */
	~DenseVariable();

	/**
 	 * Links to problem data needed to perform calculations
 	 * Calculations cannot be performed until a data object is provided
 	 * @param[in] data Pointer to the problem data
 	 */
	void LinkData(const DenseData *data);

	/** 
	 * Filles the variable with one value,
	 * i.e., x <- a * ones
	 * @param[in] a 
	 */
	void Fill(double a);

	/** 
	 * Sets the field x.y = b - A* x.z
	 */
	void InitializeConstraintMargin();

	/**
 	 * Performs the operation u <- a*x + u 
 	 * (where u is this object)
 	 * This is a level 1 BLAS operation for this object,
 	 * see http://www.netlib.org/blas/blasqr.pdf
 	 * 
 	 * @param[in] x the other variable
 	 * @param[in] a scalar
 	 *
 	 * Note that this handles the constraint margin correctly, i.e., after the
 	 * operation u.y = b - A*(u.z + a*x.z)
 	 */
	void axpy(const DenseVariable &x, double a);

	/**
 	 * Deep copy
 	 * @param[in] x variable to be copied
 	 */
	void Copy(const DenseVariable &x);

	/**
 	 * Projects the inequality duals onto the non-negative orthant,
 	 * i.e., v <- max(0,v)
 	 */
	void ProjectDuals();

	/**
	 * Compute the Euclidean norm
	 * @return ||z|| + ||v||
	 */
	double Norm() const;

	/** Accessors */
	Eigen::VectorXd& z(){ return *z_; };
	Eigen::VectorXd& v(){ return *v_; };
	Eigen::VectorXd& y(){ return *y_; }; 
	const Eigen::VectorXd& z() const { return *z_; };
	const Eigen::VectorXd& v() const { return *v_; };
	const Eigen::VectorXd& y() const { return *y_; };
	int num_constraints() { return nv_; }
	int num_variables() { return nz_; }

 private:
	int nz_ = 0; // Number of decision variable
	int nv_ = 0; // Number of inequality constraints
	const DenseData* data_ = nullptr;
	Eigen::VectorXd* z_ = nullptr; // primal variable
	Eigen::VectorXd* v_ = nullptr; // dual variable
	Eigen::VectorXd* y_ = nullptr; // inequality margin
	bool memory_allocated_ = false;

	friend class DenseResidual;
	friend class DenseLinearSolver;
	friend class DenseFeasibility;
};


}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
