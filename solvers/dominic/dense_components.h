#pragma once

#include "drake/solvers/dominic/linalg/static_matrix.h"


// contains the gereral dense qp specific classes and structures

namespace drake {
namespace solvers {
namespace dominic {


// stores the size of the qp
struct QPsize {
	int n; // primal dimension
	int q; // number of inequalities
};

// stores the problem data
class DenseData{
 public:
 	// properties *************************************
	StaticMatrix H,f,A,b;
	int n,q;
	// methods *************************************
	DenseData(double *H,double *f, double *A,double *b, QPsize size);
};

// stores primal-dual variables
class DenseVariable{
 public:
 	StaticMatrix z; // primal
	StaticMatrix v; // dual
	StaticMatrix y; // ieq margin

	DenseVariable(QPsize size);
	// 
	DenseVariable(QPsize size, double *z_mem, 
		double *v_mem, double *y_mem);
	~DenseVariable();

	// links in a DenseData object
	void LinkData(DenseData *data);
	// y <- a*ones
	void Fill(double a);
	// set the y field = b - Az
	void InitConstraintMargin();
	// y <- a*x + y
	void axpy(const DenseVariable &x, double a);
	// deep copy
	void Copy(const DenseVariable &x);
	// projects inequality duals onto the nonnegative orthant
	void ProjectDuals();


 private:
	int n,q; // sizes
	DenseData *data = nullptr; // link to the problem data
	bool y_initialized = false;
	bool memory_allocated = false;

};

// stores and computes residuals
class DenseResidual{
 public:
	StaticMatrix rz; // stationarity residual
	StaticMatrix rv; // complimentarity residual

	double alpha = 0.95; 
	// methods *************************************
	DenseResidual(QPsize size);
	~DenseResidual();

	void LinkData(DenseData *data);
	// y <- -1*y
	void Negate(); 

	// compute the penalized FB residual at (x,xbar,sigma)
	void FBresidual(const DenseVariable& x, 
		const DenseVariable& xbar, double sigma);

	// compute the natrual residual at x
	void NaturalResidual(const DenseVariable& x);
	// compute the penalized natural residuala t x
	void PenalizedNaturalResidual(const DenseVariable& x);

	// compute the norm and merit function
	double Norm();
	double Merit();
	

 private:
 	DenseData *data = nullptr; // access to the data object
 	static double pfb(double a, double b, double alpha);
 	static double max(double a, double b);
 	static double min(double a, double b);

};

// methods for solving linear systems + extra memory if needed
class DenseLinearSolver{
 public:

 	DenseLinearSolver(QPsize size);
 	~DenseLinearSolver();

 	void LinkData(const DenseData *data);
	bool Factor(const DenseVariable& x, double sigma);
	bool Solve(const DenseResidual &r, double sigma, DenseVariable *x);

	double alpha = 0.95;
	double zero_tol = 1e-13;

	
 private:
 	struct Point2D {double x; double y;};

 	DenseData *data = nullptr;
 	StaticMatrix K; // memory for the augmented Hessian

 	StaticMatrix r1; // memory for the residuals
 	StaticMatrix r2;

 	StaticMatrix Gamma; // the FB barriers/derivatives
 	StaticMatrix mus;
 	StaticMatrix gamma;


 	int n,q;

 	// computes the pfb gradient at (a,b)
 	static Point2D PFBgrad(double a, double b, double sigma);

};




}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
