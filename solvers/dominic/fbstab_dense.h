#pragma once

#include "drake/solvers/dominic/components/dense_components.h"
#include "drake/solvers/dominic/fbstab_algorithm.h"
#include "drake/solvers/dominic/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {

// a data to store input data
struct QPDataDense {
	double *H = nullptr;
	double *f = nullptr;
	double *A = nullptr;
	double *b = nullptr;
	
};

// Conveience type for the templated dense version of the algorithm
using FBstabAlgoDense = FBstabAlgorithm<DenseVariable,DenseResidual,DenseData,DenseLinearSolver,DenseFeasibilityCheck>;

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
 	SolverOut Solve(const QPDataDense &qp, double *z, double *v, double *y, bool use_initial_guess = true);

 	void UpdateOption(const char *option, double value);
 	void UpdateOption(const char *option, int value);
 	void SetDisplayLevel(FBstabAlgoDense::Display level);
 	void CheckInfeasibility(bool check);

 	// destructor
 	~FBstabDense();

 private:
 	int n = 0;
 	int q = 0;
 	FBstabAlgoDense *algo = nullptr;
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
