#pragma once

#include <Eigen/Dense>

namespace drake {
namespace solvers {
namespace fbstab {

// stores the size of the qp
struct DenseQPsize {
	int n; // primal dimension
	int q; // number of inequalities
};

// contains the gereral dense qp specific classes and structures
class DenseData{
 public:
	// fixme, this breaks if I use e.g., Vector4f or similar instead of VectorXd
	// not the end of the world but not ideal....
	DenseData(const Eigen::MatrixXd& H,const Eigen::VectorXd& f, const Eigen::MatrixXd& A,const Eigen::VectorXd& b, DenseQPsize size);

	const Eigen::MatrixXd& H_; 
 	const Eigen::VectorXd& f_;
 	const Eigen::MatrixXd& A_;
 	const Eigen::VectorXd& b_;

 private:
	int n_ = 0;
	int q_ = 0;

	friend class DenseVariable;
	friend class DenseResidual;
	friend class DenseLinearSolver;
	friend class DenseFeasibility;
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
