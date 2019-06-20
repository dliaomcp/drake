#pragma once

#include <Eigen/Dense>

namespace drake {
namespace solvers {
namespace fbstab {

// contains the gereral dense qp specific classes and structures
class DenseData{
 public:
	// fixme, this breaks if I use e.g., Vector4f or similar instead of VectorXd
	// not the end of the world but not ideal....
	DenseData(const Eigen::MatrixXd* H,const Eigen::VectorXd* f, const Eigen::MatrixXd* A,const Eigen::VectorXd* b);

	const Eigen::MatrixXd& H() const { return *H_; };
	const Eigen::VectorXd& f() const { return *f_; };
	const Eigen::MatrixXd& A() const { return *A_; }; 
	const Eigen::VectorXd& b() const { return *b_; }; 

	int num_variables() { return nz_; }
	int num_constraints() { return nv_; }

 private:
 	int nz_ = 0;
 	int nv_ = 0;
 	
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
