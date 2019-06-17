#pragma once

#include <Eigen/Dense>

#include "drake/solvers/fbstab/components/dense_data.h"


namespace drake {
namespace solvers {
namespace fbstab {

// TODO: Documentation
class DenseVariable{
 public:
	DenseVariable(DenseQPsize size); // allocates memory

	// links in a DenseData object
	void LinkData(DenseData *data);
	// x <- a*ones
	void Fill(double a);
	// set the y field = b - Az
	void InitializeConstraintMargin();
	// y <- a*x + y
	void axpy(const DenseVariable &x, double a);
	// deep copy
	void Copy(const DenseVariable &x);
	// projects inequality duals onto the nonnegative orthant
	void ProjectDuals();
	double Norm() const;

	Eigen::VectorXd& z(){ return z_; };
	Eigen::VectorXd& v(){ return v_; };
	Eigen::VectorXd& y(){ return y_; }; 

	friend std::ostream &operator<<(std::ostream& output, const DenseVariable &x);

 private:
	int n_,q_; // sizes
	DenseData *data_ = nullptr;

	Eigen::VectorXd z_; // primal variable
	Eigen::VectorXd v_; // dual variable
	Eigen::VectorXd y_; // inequality margin

	friend class DenseResidual;
	friend class DenseLinearSolver;
	friend class DenseFeasibility;
};


}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
