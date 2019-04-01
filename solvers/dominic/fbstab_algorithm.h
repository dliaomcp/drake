#pragma once

#include <cmath>
#include <cstdio>

#include "drake/solvers/dominic/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {

// return values for the solver
enum ExitFlag {
	SUCCESS = 0,
	DIVERGENCE = 1,
	MAXITERATIONS = 2,
	INFEASIBLE = 3,
	UNBOUNDED_BELOW = 4
};

// container for outputs
struct SolverOut {
	ExitFlag eflag;
	double residual;
	int newton_iters;
	int prox_iters;
};


// this class implements the FBstab algorithm
template <class Variable, class Residual, class Data, class LinearSolver, class Feasibility>
class FBstabAlgorithm{
 public:
	// display settings
	enum Display {
		OFF = 0, // no display
		FINAL = 1, // prints message upon completion
		ITER = 2, // basic information at each outer loop iteration
		ITER_DETAILED = 3 // print inner loop information
	};

	FBstabAlgorithm::Display display_level = ITER_DETAILED;

	// initializes the component objects needed by the solver
	FBstabAlgorithm(Variable *x1, Variable *x2, 
		Variable *x3, Variable *x4, Residual *r1, Residual *r2, LinearSolver *lin_sol, Feasibility *fcheck);

	// run the solve routine for given problem data and initial guess
	SolverOut Solve(Data *qp_data, Variable *x0);

	void UpdateOption(const char *option, double value);
	void UpdateOption(const char *option, int value);
	void UpdateOption(const char *option, bool value);
	void DeleteComponents();

 private:
	enum { kNonmonotoneLinesearch = 3 }; 
	double merit_values[kNonmonotoneLinesearch] = { 0.0 };

	enum InfeasibilityStatus {
		FEASIBLE = 0,
		PRIMAL = 1,
		DUAL = 2,
		BOTH = 3
	};

 	InfeasibilityStatus CheckInfeasibility(const Variable &x);

 	// shifts all elements up one, inserts x at 0
	static void ShiftAndInsert(double *buffer, double x, int buff_size);
	static void ClearBuffer(double *buffer, int buff_size);
	// returns the largest element in the vector
	static double VectorMax(double* vec, int length);

	// elementwise min and max
	template <class T>
	static T max(T a, T b){
		return (a>b) ? a : b;
	}

	template <class T>
	static T min(T a, T b){
		return (a>b) ? b : a;
	}

	// printing
	void PrintIterHeader();
	void PrintIterLine(int prox_iters, int newton_iters, const Residual &r, const Residual &r_inner,double itol);
	void PrintDetailedHeader(int prox_iters, int newton_iters, const Residual &r);
	void PrintDetailedLine(int iter, double step_length, const Residual &r);
	void PrintDetailedFooter(double tol, const Residual &r);
	void PrintFinal(int prox_iters, int newton_iters, ExitFlag eflag, const Residual &r);

	static bool strcmp(const char *x, const char *y);
	
	// variable objects
	Variable *xk = nullptr;
	Variable *xi = nullptr;
	Variable *xp = nullptr;
	Variable *dx = nullptr;

	// residual objects
	Residual *rk = nullptr;
	Residual *ri = nullptr;

	// linear system solver object
	LinearSolver *linear_solver = nullptr;
	Feasibility *feas = nullptr;

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
	int max_newton_iters = 20;
	int max_prox_iters = 30;
	int max_inner_iters = 50;
	int max_linesearch_iters = 20;

	bool check_feasibility = true;

};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake

#include "drake/solvers/dominic/fbstab_algorithm_impl.h"