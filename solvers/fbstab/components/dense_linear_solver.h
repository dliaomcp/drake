#pragma once

#include <Eigen/Dense>

#include "drake/solvers/fbstab/components/dense_variable.h"
#include "drake/solvers/fbstab/components/dense_residual.h"
#include "drake/solvers/fbstab/components/dense_data.h"


// contains the gereral dense qp specific classes and structures

namespace drake {
namespace solvers {
namespace fbstab {

// methods for solving linear systems + extra memory if needed
class DenseLinearSolver{
 public:
 	struct Point2D {double x; double y;};

 	DenseLinearSolver(DenseQPsize size);

 	void LinkData(DenseData *data);
 	void SetAlpha(double alpha);
 	
	bool Factor(const DenseVariable &x, const DenseVariable &xbar, double sigma);
	bool Solve(const DenseResidual &r, DenseVariable *x);

	Eigen::MatrixXd& K(){ return K_; };
	
 private:
 	int n_,q_;
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

 	DenseData *data_ = nullptr;
 	// computes the PFB function gradient at (a,b)
 	Point2D PFBGradient(double a, double b);

 	void CholeskySolve(const Eigen::MatrixXd& A, Eigen::VectorXd* b);
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
