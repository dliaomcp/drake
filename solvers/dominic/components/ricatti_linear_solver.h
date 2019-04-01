#pragma once

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
 	void SetAlpha(double alpha);
 	bool Factor(const MPCVariable &x, const MPCVariable &xbar, double sigma);
 	bool Solve(const MPCResidual &r, MPCVariable *dx);

 	// barrier augmented cost matrices
 	MatrixSequence Q_; 
 	MatrixSequence R_;
 	MatrixSequence S_;

 	// Storage for the matrix portion of the recursion
 	MatrixSequence P_;
 	MatrixSequence SG_;
 	MatrixSequence M_;
 	MatrixSequence L_;
 	MatrixSequence SM_;
 	MatrixSequence AM_;

 	// Storage for the vector portion of the recursion
 	MatrixSequence h_;
 	MatrixSequence th_;

 	// Storage for barrier terms
 	StaticMatrix gamma_; 
 	StaticMatrix mus_;
 	StaticMatrix Gamma_;

 	// Workspace matrices
 	StaticMatrix Etemp_;
 	StaticMatrix Linv_;

 	// Workspace vectors
 	StaticMatrix tx_;
 	StaticMatrix tl_;
 	StaticMatrix tu_;
 	StaticMatrix r1_;
 	StaticMatrix r2_;

 private:
 	int N_, nx_, nu_, nc_;
 	int nz_, nl_, nv_;

 	MPCData *data_ = nullptr;
 	double zero_tol_ = 1e-13;
 	bool memory_allocated_ = false;
 	double alpha_ = 0.95;

 	Point2D PFBgrad(double a, double b, double sigma);
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
