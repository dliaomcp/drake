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

// This structure stores the input data.
struct DenseQPData {
	const Eigen::MatrixXd* H = nullptr;
	const Eigen::MatrixXd* A = nullptr;
	const Eigen::VectorXd* f = nullptr;
	const Eigen::VectorXd* b = nullptr;
};

struct DenseQPVariable {
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
 	FBstabDense(int num_variables, int num_constraints);
 	~FBstabDense();

 	// Solve an instance of the QP
 	// Inputs are the QP data
 	SolverOut Solve(const DenseQPData& qp, const DenseQPVariable& x, bool use_initial_guess = true);

 	void UpdateOption(const char *option, double value);
 	void UpdateOption(const char *option, int value);
 	void UpdateOption(const char *option, bool value);
 	void SetDisplayLevel(FBstabAlgoDense::Display level);

 private:
 	int n_ = 0;
 	int q_ = 0;
 	FBstabAlgoDense *algorithm_ = nullptr;
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
