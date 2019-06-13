#pragma once

#include <cmath>
#include <cstdio>

#include "drake/solvers/fbstab/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {

/**
 * Return codes for the solver.
 */
enum ExitFlag {
	SUCCESS = 0,
	DIVERGENCE = 1,
	MAXITERATIONS = 2,
	INFEASIBLE = 3,
	UNBOUNDED_BELOW = 4
};

/**
 * Packages the exit flag, overall residual, 
 * and iteration counts.
 */
struct SolverOut {
	ExitFlag eflag;
	double residual;
	int newton_iters;
	int prox_iters;
};

/**
 * FBstabAlgorithm implements the FBstab solver for 
 * convex quadratic programs. 
 *
 * FBstab tries to solve the following convex QP 
 * 
 * min.  1/2 z'*H*z + f'*z
 * 
 * s.t.  Gz =  h
 *       Az <= b
 *
 * The algorithm is implemented using to abstract objects 
 * representing variables, residuals etc. 
 * These are template parameters for the class and
 * should be written so as to be efficient for specific classes
 * of QPs.
 * 
 * @tparam Variable storage and methods for working with primal-dual variables 
 * @tparam Residual storage and methods for computing QP residuals
 * @tparam Data QP type specific data storage
 * @tparam LinearSolver solves linear systems of equations
 * @tparam Feasibility Checks for primal-dual infeasibility
 */
template <class Variable, class Residual, class Data, class LinearSolver, class Feasibility>
class FBstabAlgorithm{
 public:
	/**
	 * Display settings
	 */
	enum Display {
		OFF = 0, // no display
		FINAL = 1, // prints message upon completion
		ITER = 2, // basic information at each outer loop iteration
		ITER_DETAILED = 3 // print inner loop information
	};

	/**
	 * Saves the components objects needed by the solver.
	 * 
	 * @param[in] x1,x2,x3,x4 Variable objects used by the solver
	 * @param[in] r1,r2 Residual objects used by the solver
	 * @param[in] lin_sol Linear solver used by the solver
	 * @param[in] fcheck Feasibility checker used by the solver
	 */
	FBstabAlgorithm(Variable *x1, Variable *x2, 
		Variable *x3, Variable *x4, Residual *r1, Residual *r2, LinearSolver *lin_sol, Feasibility *fcheck);

	/**
	 * Attempts to solve the QP for the given
	 * data starting from the supplied initial guess
	 * 
	 * @param[in] qp_data Problem data 
	 * @param[both] x0 Initial primal-dual guess, overwritten with the solution
	 * 
	 * @return Details on the solver output
	 */
	SolverOut Solve(Data *qp_data, Variable *x0);

	/**
	 * Allows setting of algorithm options
	 * @param[in] option Option name
	 * @param[in] value New value
	 *
	 * Possible options and default parameters are:
	 * sigma0{1e-8}: Initial stabilization parameter
	 * alpha{0.95}: Penalized FB function parameter
	 * beta{0.7}: Backtracking linesearch parameter
	 * eta{1e-8}: Sufficient decrease parameter
	 * inner_tol_multiplier{0.2}: Reduction factor for subproblem tolerance
	 *
	 * The algorithm exists when: ||F(x)|| <= abs_tol + ||F(x0)|| rel_tol
	 * abs_tol{1e-6}: Absolute tolerance
	 * rel_tol{1e-12}: Relative tolerance
	 * stall_tol{1e-10}: If the residual doesn't decrease by at least this failure
	 * is declared
	 * infeas_tol{1e-8}: Relative tolerance used in feasibility checking
	 *
	 * inner_tol_max{1.0}: Maximum value for the subproblem tolerance
	 * inner_tol_min{1e-12}: Minimum value for the subproblem tolerance
	 *
	 * max_newton_iters{200}: Maximum number of Newton iterations before timeout
	 * max_prox_iters{30}: Maximum number of proximal iterations before timeout
	 * max_inner_iters{50}: Maximum number of iterations that can be applied
	 * to a single subproblem
	 * max_linesearch_iters{20}: Maximum number of backtracking linesearch steps
	 *
	 * check_feasibility{true}: Enable or disable the feasibility checker
	 * if the problem is known to be feasible then it should be disabled for speed
	 */
	void UpdateOption(const char *option, double value);
	void UpdateOption(const char *option, int value);
	void UpdateOption(const char *option, bool value);
	Display& display_level(){ return display_level_ ;}

	/**
	 * Deletes all the component objects. CALLS DELETE USE WITH CARE.
	 */
	void DeleteComponents();

 private:
	enum { kNonmonotoneLinesearch = 3 }; 
	double merit_values_[kNonmonotoneLinesearch] = { 0.0 };

	/**
	 * Codes for infeasibility detection
	 */
	enum InfeasibilityStatus {
		FEASIBLE = 0,
		PRIMAL = 1,
		DUAL = 2,
		BOTH = 3
	};

	/**
	 * Attempts to solve a proximal subproblem x = P(xbar,sigma) using
	 * the semismooth Newton's method. See (11) in https://arxiv.org/pdf/1901.04046.pdf
	 * 
	 * @param[both]  x      Initial guess, overwritten with the solution.
	 * @param[in]    xbar   Current proximal (outer) iterate
	 * @param[in]    tol    Desired tolerance for the inner residual
	 * @param[in]    sigma  Regularization strength
	 * @param[in]    Eouter Current overall problem residual
	 * @return       Residual for the outer problem evaluated at x
	 *
	 *
	 * Note: Uses
	 * rk,ri,dx, and xp as workspaces
	 * 
	 */
	double SolveSubproblem(Variable *x,Variable *xbar, double tol, double sigma, double Eouter);

	/**
	 * Checks if the primal-dual QP pair is feasible
	 * @param[in]  x Point at which to check for infeasibility
	 * @return   Feasibility status of primal and dual problems
	 */
 	InfeasibilityStatus CheckInfeasibility(const Variable &x);

 	/**
 	 * Shifts all elements in an array up 1 spot and adds a new element at 0
 	 * @param[out] buffer    Pointer to the array
 	 * @param[in] x         value to be inserted at buffer[0]
 	 * @param[in] buff_size length of the array
 	 */
	static void ShiftAndInsert(double *buffer, double x, int buff_size);
	/**
	 * Sets all elements to 0
	 * @param[out] buffer    Pointer to array
	 * @param[in] buff_size length of the array
	 */
	static void ClearBuffer(double *buffer, int buff_size);
	
	/**
	 * Returns the maximum value in an array
	 * @param[in]  vec    pointer to the array
	 * @param[in]  length length
	 * @return        maximum value
	 */
	static double VectorMax(double* vec, int length);

	/**
	 * Element wise max
	 */
	template <class T>
	static T max(T a, T b){
		return (a>b) ? a : b;
	}
	/**
	 * Element wise min
	 */
	template <class T>
	static T min(T a, T b){
		return (a>b) ? b : a;
	}

	/**
	 * Compares two C style strings
	 * @param[in]  x string 1
	 * @param[in]  y string 2
	 * @return   true is x and y are equal
	 */
	static bool strcmp(const char *x, const char *y);

	/**
	 * A collection of functions for printing optimization progress to 
	 * stdout
	 */
	void PrintIterHeader();
	void PrintIterLine(int prox_iters, int newton_iters, const Residual &rk, const Residual &ri,double itol);
	void PrintDetailedHeader(int prox_iters, int newton_iters, const Residual &r);
	void PrintDetailedLine(int iter, double step_length, const Residual &r);
	void PrintDetailedFooter(double tol, const Residual &r);
	void PrintFinal(int prox_iters, int newton_iters, ExitFlag eflag, const Residual &r);

	FBstabAlgorithm::Display display_level_ = FINAL;

	// iteration counters
	int newton_iters_ = 0;
	int prox_iters_ = 0;

	// variable objects
	Variable *xk_ = nullptr; // outer loop variable
	Variable *xi_ = nullptr; // inner loop variable
	Variable *xp_ = nullptr; // workspace
	Variable *dx_ = nullptr; // workspace

	// residual objects
	Residual *rk_ = nullptr; // outer loop residual
	Residual *ri_ = nullptr; // inner loop residual

	// linear system solver object
	LinearSolver *linear_solver_ = nullptr;
	Feasibility *feasibility_ = nullptr;

	// Algorithm parameters
	// See https://arxiv.org/pdf/1901.04046.pdf
	double sigma0_ = 1e-8;
	double alpha_ = 0.95;
	double beta_ = 0.7;
	double eta_ = 1e-8;
	double inner_tol_multiplier_ = 1.0/5;

	// tolerances
	double abs_tol_ = 1e-6;
	double rel_tol_ = 1e-12;
	double stall_tol_ = 1e-10;
	double infeasibility_tol_ = 1e-8;

	// tolerance guards
	double inner_tol_max_ = 1.0;
	double inner_tol_min_ = 1e-12;
	
	// maximum iterations
	int max_newton_iters_ = 200;
	int max_prox_iters_ = 30;
	int max_inner_iters_ = 50;
	int max_linesearch_iters_ = 20;

	bool check_feasibility_ = true;

};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake

#include "drake/solvers/fbstab/fbstab_algorithm-inl.h"