#pragma once

#include <Eigen/Dense>

#include "drake/solvers/fbstab/components/dense_data.h"

namespace drake {
namespace solvers {
namespace fbstab {

// TODO: Documentation
class DenseVariable{
 public:
	DenseVariable(int nz, int nv); // allocates memory
	DenseVariable(Eigen::VectorXd* z, Eigen::VectorXd* v, Eigen::VectorXd* y);
	~DenseVariable();
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

	Eigen::VectorXd& z(){ return *z_; };
	Eigen::VectorXd& v(){ return *v_; };
	Eigen::VectorXd& y(){ return *y_; }; 

	const Eigen::VectorXd& z() const { return *z_; };
	const Eigen::VectorXd& v() const { return *v_; };
	const Eigen::VectorXd& y() const { return *y_; };

	int num_constraints() { return nv_; }
	int num_variables() { return nz_; }

 private:
	int nz_ = 0; // number of decision variable
	int nv_ = 0; // number of inequality constraints
	DenseData* data_ = nullptr;

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
