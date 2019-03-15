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

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
