#pragma once

#include "drake/solvers/dominic/dense_components.h"
#include "drake/solvers/dominic/fbstab_algorithm.h"
#include "drake/solvers/dominic/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {

// a data to store input data
struct QPData {
	double *H = nullptr;
	double *f = nullptr;
	double *A = nullptr;
	double *b = nullptr;
	
};

// the main object for the C++ API
class FBstabDense {

 public:
 	// dynamically initializes component classes 
 	// n: number of decision variables
 	// q: number of constraints
 	FBstabDense(int n,int q);

 	// Solve an instance of the QP
 	// Inputs are the QP data
 	// z is a pointer to e.g., new double[n]
 	// v,y are pointer to e.g., new double[q]
 	// the solution is stored in z and v with y = b - Az
 	ExitFlag Solve(const QPdata &qp, double *z, double *v, double *y, bool use_initial_guess = true);

 	// destructor
 	~FBstabDense();

 private:
 	int n;
 	int q;
}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
