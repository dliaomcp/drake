#pragma once

#include "drake/solvers/dominic/dense_components.h"
#include "drake/solvers/dominic/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {

// return values for the solver
enum ExitFlag {
	SUCCESS = 0;
	DIVERGENCE = 1;
	MAXITERATIONS = 2;
	PRIMAL_INFEASIBLE = 3;
	UNBOUNDED_BELOW = 4;
};

// container for outputs
struct SolverOut {
	ExitFlag eflag;
	double residual;
	int newton_iters;
	int prox_iters;
};

// TODO: Make me a template class later
// this class implements the FBstab algorithm
class FBstabAlgorithm{
public:
	
	// display settings
	enum Display {
		OFF = 0; // no display
		FINAL = 1; // prints message upon completion
		ITER = 2; // basic information at each outer loop iteration
		ITER_DETAILED = 3; // print inner loop information
	};

	// initializes the component objects needed by the solver
	FBstabAlgorithm(DenseVariable *x1, DenseVariable *x2, 
		DenseVariable *x3, DenseVariable *x4, DenseResidual *r1, DenseResidual *r2, DenseLinearSolver *linear_solver);

	void UpdateOptions();

	// run the solve routine for a given problem data
	// the x0 passed to the inner solver 
	// ABSOLUTELY MUST have its y field initialized
	SolverOut Solve(const DenseData &qp_data, DenseVariable *x0);

	// destructor
	~FBstabAlgorithm();

private:
	enum { kNonmonotoneLineseach = 5 }; 
	double merit_values[kNonmonotoneLineseach] = { 0.0 };

 	int CheckInfeasibility(const DenseVariable& dx);

	void ShiftAndInsert(double* buffer, double x, int buff_size);

	static double VectorMax(double* vec, int length);
	static double max(double a,double b);
	static double min(double a,double b);

	// problem data
	DenseData *data = nullptr;

	// variable objects
	DenseVariable *xk;
	DenseVariable *xi;
	DenseVariable *xp;
	DenseVariable *dx;

	// residual objects
	DenseResidual *rk;
	DenseResidual *ri;

	// linear system solver object
	DenseResidual *linear_solver;

	// tolerances
	double abs_tol = 1e-6;
	double rel_tol = 1e-12;
	double stall_tol = 1e-10;
	double infeas_tol = 1e-8;

	// algorithm parameters
	double sigma0 = 1e-8;
	double alpha = 0.95;
	double beta = 0.7;
	double eta = 1e-8;
	double inner_tol_multiplier = 1.0/5;

	// tolerance guards
	double inner_tol_max = 1.0;
	double inner_tol_min = 1e-13;
	
	// maximum iterations
	int max_newton_iters = 100;
	int max_prox_iters = 30;
	int max_inner_iters = 50;
	int max_linesearch_iters = 20;

	bool check_infeasibility = true;

};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake