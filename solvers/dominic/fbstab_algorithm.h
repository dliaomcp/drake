#pragma once

#include "drake/solvers/dominic/dense_components.h"
#include "drake/solvers/dominic/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {

// return values for the solver
enum ExitFlag {
	SUCCESS = 0,
	DIVERGENCE = 1,
	MAXITERATIONS = 2,
	PRIMAL_INFEASIBLE = 3,
	UNBOUNDED_BELOW = 4
};

// container for outputs
struct SolverOut {
	ExitFlag eflag;
	double residual;
	int newton_iters;
	int prox_iters;
};

// TODO: Make me a template class 
// this class implements the FBstab algorithm
class FBstabAlgorithm{
public:
	bool check_infeasibility = true;

	// display settings
	enum Display {
		OFF = 0, // no display
		FINAL = 1, // prints message upon completion
		ITER = 2, // basic information at each outer loop iteration
		ITER_DETAILED = 3 // print inner loop information
	};

	FBstabAlgorithm::Display display_level = ITER_DETAILED;

	// initializes the component objects needed by the solver
	FBstabAlgorithm(DenseVariable *x1, DenseVariable *x2, 
		DenseVariable *x3, DenseVariable *x4, DenseResidual *r1, DenseResidual *r2, DenseLinearSolver *lin_sol);

	// run the solve routine for a given problem data
	SolverOut Solve(DenseData *qp_data, DenseVariable *x0);

	void UpdateOption(const char *option, double value);
	void UpdateOption(const char *option, int value);
	void DeleteComponents();

private:
	enum { kNonmonotoneLinesearch = 3 }; 
	double merit_values[kNonmonotoneLinesearch] = { 0.0 };

	enum InfeasibilityStatus {
		FEASIBLE = 0,
		INFEASIBLE = 1,
		UNBOUNDED_BELOW = 2
	};

 	InfeasibilityStatus CheckInfeasibility(const DenseVariable& dx);

 	// shifts all elements up one, inserts x at 0
	static void ShiftAndInsert(double *buffer, double x, int buff_size);
	static void ClearBuffer(double *buffer, int buff_size);
	// returns the largest element in the vector
	static double VectorMax(double* vec, int length);

	// elementwise min and max
	static double max(double a,double b);
	static double min(double a,double b);

	static int max(int a,int b);
	static int min(int a,int b);

	// printing
	void IterHeader();
	void IterLine(int prox_iters,int newton_iters,const DenseResidual &r);
	void DetailedHeader(int prox_iters, int newton_iters, const DenseResidual &r);
	void DetailedLine(int iter, double step_length, const DenseResidual &r);
	void DetailedFooter(double tol, const DenseResidual &r);
	void PrintFinal(int prox_iters, int newton_iters, ExitFlag eflag, const DenseResidual &r);

	static bool strcmp(const char *x, const char *y);
	
	// problem data
	DenseData *data = nullptr;

	// variable objects
	DenseVariable *xk = nullptr;
	DenseVariable *xi = nullptr;
	DenseVariable *xp = nullptr;
	DenseVariable *dx = nullptr;

	// residual objects
	DenseResidual *rk = nullptr;
	DenseResidual *ri = nullptr;

	// linear system solver object
	DenseLinearSolver *linear_solver = nullptr;

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
	double inner_tol_min = 1e-12;
	
	// maximum iterations
	int max_newton_iters = 100;
	int max_prox_iters = 30;
	int max_inner_iters = 50;
	int max_linesearch_iters = 20;

};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake