#pragma once

#include <Eigen/Dense>

namespace drake {
namespace solvers {
namespace fbstab {


/** 
 * Represents data for quadratic programing problems of the following type (1):
 * 
 * min.    1/2  z'Hz + f'z
 * s.t.         Az <= b
 * 
 * where H is symmetric and positive semidefinite. 
 */
class DenseData{
 public:
 	/**
 	 * Store the problem data and perform input validation.
 	 * This class assumes that the pointers to the data remain valid.
 	 * 
 	 * @param[in] H Hessian matrix
 	 * @param[in] f Linear term
 	 * @param[in] A Constraint matrix
 	 * @param[in] b Constraint vector
 	 */
	DenseData(const Eigen::MatrixXd* H,const Eigen::VectorXd* f, const Eigen::MatrixXd* A,const Eigen::VectorXd* b);

	/** 
	 * Read only accessors.
	 */
	const Eigen::MatrixXd& H() const { return *H_; };
	const Eigen::VectorXd& f() const { return *f_; };
	const Eigen::MatrixXd& A() const { return *A_; }; 
	const Eigen::VectorXd& b() const { return *b_; }; 

	/**
	 * @return number of decision variables (i.e., dimension of z)
	 */
	int num_variables() { return nz_; }
	/** 
	 * @return number of inequality constraints
	 */
	int num_constraints() { return nv_; }

 private:
 	int nz_ = 0; // Number of decision variables.
 	int nv_ = 0; // Number of constraints.
 	
 	const Eigen::MatrixXd* H_; 
 	const Eigen::VectorXd* f_;
 	const Eigen::MatrixXd* A_;
 	const Eigen::VectorXd* b_;

	friend class DenseVariable;
	friend class DenseResidual;
	friend class DenseLinearSolver;
	friend class DenseFeasibility;
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
