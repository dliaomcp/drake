#pragma once

#include "drake/solvers/dominic/linalg/static_matrix.h"
#include "drake/solvers/dominic/linalg/matrix_sequence.h"
#include "drake/solvers/dominic/components/mpc_data.h"

namespace drake {
namespace solvers {
namespace fbstab {


/**
 * Implements primal-dual variables for model predictive
 * control QPs. See mpc_data.h for the mathematical description.
 * Stores variables and defines methods implementing useful operations.
 * 
 */
class MPCVariable{
 public:

 	/**
 	 * Primal-dual variables have 4 fields:
 	 * z: Decision variables (x0,u0,x1,u1, ... xN,uN)
 	 * l: Co-states/ equality duals (l0, ... ,lN)
 	 * v: Inequality duals (v0, ..., vN)
 	 * y: Inequality margins (y0, ..., yN)
 	 *
 	 * length(z) = (nx*nu)*(N+1)
 	 * length(l) = nx*(N+1)
 	 * length(v) = nc*(N+1)
 	 * length(y) = nc*(N+1)
 	 * 
 	 */
 	StaticMatrix z_;
 	StaticMatrix l_; 
 	StaticMatrix v_; 
 	StaticMatrix y_; 

 	/**
 	 * Allocates memory for a primal-dual variable.
 	 *
 	 * @param[in] size Problem size
 	 * 
 	 * size has the following fields
 	 * N:  horizon length
 	 * nx: number of states
 	 * nu: number of control input
 	 * nc: number of constraints per stage
 	 */
 	MPCVariable(QPsizeMPC size);

 	/**
 	 * Creates a primal-dual variable using preallocated memory.
 	 *
 	 * @param[in] size Problem size
 	 * @param[in] z    Memory for the decision variables. Overwritten.
 	 * @param[in] l    Memory for the equality duals.     Overwritten.
 	 * @param[in] v    Memory for the inequality duals.   Overwritten.
 	 * @param[in] y    Memory for the constraint margins. Overwritten.
 	 *
 	 * size has the following fields
 	 * N:  horizon length
 	 * nx: number of states
 	 * nu: number of control input
 	 * nc: number of constraints per stage
 	 * 
 	 * Ensure that:
 	 * length(z) = (nx*nu)*(N+1)
 	 * length(l) = nx*(N+1)
 	 * length(v) = nc*(N+1)
 	 * length(y) = nc*(N+1)
 	 *
 	 * length(x) = sizeof(x)/sizeof(double)
 	 *
 	 */
 	MPCVariable(QPsizeMPC size, double *z, double *l, double *v, double *y);

 	/**
 	 * Frees allocated memory.
 	 */
 	~MPCVariable();

 	/**
 	 * Links to problem data needed to perform calculations
 	 * Calculations cannot be performed until a data object is provided
 	 * @param[in] data Pointer to the problem data
 	 */
 	void LinkData(MPCData *data);

 	/**
 	 * Fills the variable with all a
 	 * @param[in] a Value to fill
 	 */
 	void Fill(double a);

 	/**
 	 * Sets the constraint margin to y = b - Az
 	 */
 	void InitConstraintMargin();

 	/**
 	 * Sets u <- a*x + u (where u is this object)
 	 * @param[in] x vector
 	 * @param[in] a scalar
 	 *
 	 * Note that this handles the constraint margin correctly, i.e., after the
 	 * operation u.y = b - A*(u.z + a*x.z)
 	 */
 	void axpy(const MPCVariable &x, double a);

 	/**
 	 * (Deep) copies x into this
 	 * @param[in] x variable to be copied
 	 */
 	void Copy(const MPCVariable &x);

 	/**
 	 * Projects the inequality duals onto the non-negative orthant,
 	 * i.e., v <- max(0,v)
 	 */
 	void ProjectDuals();

 	/**
 	 * Computes the Euclidean norm 
 	 * @return ||z|| + ||l|| + ||v||
 	 */
 	double Norm();

 	/**
 	 * Computes the infinity norm
 	 * @return ||z|| + ||l|| + ||v||
 	 */
 	double InfNorm();

 	/**
 	 * Allows for writing to a stream, i.e., std::cout << *this
 	 */
 	friend std::ostream &operator<<(std::ostream& output, const MPCVariable &x);

 private:
 	int N_, nx_, nu_, nc_;
 	int nz_, nl_, nv_;
 	bool memory_allocated_ = false;
 	MPCData *data_ = nullptr;
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake