#pragma once

#include "drake/solvers/fbstab/components/dense_data.h"
#include "drake/solvers/fbstab/components/dense_variable.h"
#include "drake/solvers/fbstab/components/dense_residual.h"
#include "drake/solvers/fbstab/components/dense_linear_solver.h"
#include "drake/solvers/fbstab/components/dense_feasibility.h"
#include "drake/solvers/fbstab/fbstab_algorithm.h"
#include "drake/solvers/fbstab/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {

// a data to store input data
struct DenseQPData {
	double *H = nullptr;
	double *f = nullptr;
	double *A = nullptr;
	double *b = nullptr;
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
 	SolverOut Solve(const DenseQPData &qp, double *z, double *v, double *y, bool use_initial_guess = true);

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
