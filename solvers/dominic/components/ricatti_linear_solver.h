#pragma once

#include <cmath>

#include "drake/solvers/dominic/linalg/static_matrix.h"
#include "drake/solvers/dominic/linalg/matrix_sequence.h"
#include "drake/solvers/dominic/components/mpc_data.h"
#include "drake/solvers/dominic/components/mpc_variable.h"
#include "drake/solvers/dominic/components/mpc_residual.h"

namespace drake {
namespace solvers {
namespace fbstab {


class RicattiLinearSolver{
 public:
 	struct Point2D {double x; double y;};

 	RicattiLinearSolver(QPsizeMPC size);
 	~RicattiLinearSolver();

 	void LinkData(MPCData *data);
 	bool Factor(const MSVariable &x, const MSVariable &xbar, double sigma);
 	bool Solve(const MSResidual &r, MSVariable *dx);

 private:
 	int N_, nx_, nu_, nc_;
 	int nz_, nl_, nv_;

 	double alpha_ = 0.95;
 	MPCData *data_ = nullptr;
 	double zero_tol_ = 1e-13;

 	MatrixSequence Q_, R_, S_;
 	MatrixSequence P_;
 	MatrixSequence Sig_;
 	MatrixSequence M_;
 	MatrixSequence L_;
 	MatrixSequence SM_;
 	MatrixSequence AM_;

 	MatrixSequence h_;
 	MatrixSequence theta_;

 	Point2D PFBgrad(double a, double b, double sigma);
};

DenseLinearSolver::Point2D DenseLinearSolver::PFBgrad(double a,
 double b, double sigma){
	double y = 0;
	double x = 0;
	double r = sqrt(a*a + b*b);
	double d = 1.0/sqrt(2.0);

	if(r < zero_tol){
		x = alpha*(1.0-d);
		y = alpha*(1.0-d);

	} else if((a > 0) && (b > 0)){
		x = alpha * (1.0- a/r) + (1.0-alpha) * b;
		y = alpha * (1.0- b/r) + (1.0-alpha) * a;

	} else {
		x = alpha * (1.0 - a/r);
		y = alpha * (1.0 - b/r);
	}

	Point2D out = {x, y};
	return out;
}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
