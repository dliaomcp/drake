#pragma once

#include <Eigen/Dense>

#include "drake/solvers/fbstab/components/dense_data.h"
#include "drake/solvers/fbstab/components/dense_variable.h"
#include "drake/solvers/fbstab/components/dense_residual.h"
#include "drake/solvers/fbstab/components/dense_linear_solver.h"
#include "drake/solvers/fbstab/components/dense_feasibility.h"
#include "drake/solvers/fbstab/fbstab_algorithm.h"

namespace drake {
namespace solvers {
namespace fbstab {

// a data to store input data
// these should be
struct DenseQPData {
	const Eigen::MatrixXd& H;
	const Eigen::MatrixXd& A;
	const Eigen::VectorXd& f;
	const Eigen::VectorXd& b;

	DenseQPData(const Eigen::MatrixXd& H_, const Eigen::MatrixXd& A_, const Eigen::VectorXd& f, const Eigen::VectorXd& b)
	: H(H_), A(A_), f(f_), b(b_) {}
};

struct DenseQPSolution {
	Eigen::VectorXd* z = nullptr;
	Eigen::VectorXd* v = nullptr;
	Eigen::VectorXd* y = nullptr;
};

// Conveience type for the templated dense version of the algorithm
using FBstabAlgoDense = FBstabAlgorithm<DenseVariable,DenseResidual,DenseData,DenseLinearSolver,DenseFeasibility>;

// the main object for the C++ API
class FBstabDense {

 public:
 	// dynamically initializes component classes 
 	// n: number of decision variables
 	// q: number of constraints
 	FBstabDense(int n, int q);

 	// Solve an instance of the QP
 	// Inputs are the QP data
 	// z = new double[n]
 	// v,y = new double[q]
 	// the solution is stored in z and v with y = b - Az
 	SolverOut Solve(const DenseQPData &qp, const DenseQPSolution& x, bool use_initial_guess = true);

 	~FBstabDense();
 	
 	void UpdateOption(const char *option, double value);
 	void UpdateOption(const char *option, int value);
 	void UpdateOption(const char *option, bool value);
 	void SetDisplayLevel(FBstabAlgoDense::Display level);


 private:
 	int n_ = 0;
 	int q_ = 0;
 	FBstabAlgoDense *algo = nullptr;
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
