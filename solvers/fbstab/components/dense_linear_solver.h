#pragma once

#include "drake/solvers/fbstab/linalg/static_matrix.h"
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
 	~DenseLinearSolver();

 	void LinkData(DenseData *data);
 	void SetAlpha(double alpha);
 	
	bool Factor(const DenseVariable &x, const DenseVariable &xbar, double sigma);
	bool Solve(const DenseResidual &r, DenseVariable *x);

	
 private:
 	int n_,q_;
 	double alpha_ = 0.95;
	double zero_tolerance_ = 1e-13;

	// workspace variables
	StaticMatrix K_; // memory for the augmented Hessian
	StaticMatrix r1_; // residual
 	StaticMatrix r2_; // residual

 	StaticMatrix Gamma_;
 	StaticMatrix mus_;
 	StaticMatrix gamma_;

 	DenseData *data_ = nullptr;

 	// computes the PFB function gradient at (a,b)
 	Point2D PFBGradient(double a, double b, double sigma);
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
