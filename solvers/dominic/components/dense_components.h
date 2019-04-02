#pragma once

#include "drake/solvers/dominic/linalg/static_matrix.h"


// contains the gereral dense qp specific classes and structures

namespace drake {
namespace solvers {
namespace fbstab {


// stores the size of the qp
// rename to DenseQPsize?
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
	DenseVariable(QPsize size, double *z_mem, 
		double *v_mem, double *y_mem);
	~DenseVariable();

	// links in a DenseData object
	void LinkData(DenseData *data_);
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
	double Norm();
	double InfNorm();

	friend std::ostream &operator<<(std::ostream& output, const DenseVariable &x);


 private:
	int n,q; // sizes
	DenseData *data = nullptr; // link to the problem data
	bool memory_allocated = false;

	friend class DenseFeasibilityCheck;
};

// stores and computes residuals
class DenseResidual{
 public:
	StaticMatrix rz; // stationarity residual
	StaticMatrix rv; // complimentarity residual

	double alpha = 0.95; 
	double z_norm = 0.0;
	double v_norm = 0.0;
	double l_norm = 0.0;
	// methods *************************************
	DenseResidual(QPsize size);
	~DenseResidual();

	void LinkData(DenseData *data);
	void SetAlpha(double alpha_);
	void Negate(); // y <- -1*y

	// compute the penalized FB residual at (x,xbar,sigma)
	void FBresidual(const DenseVariable& x, 
		const DenseVariable& xbar, double sigma);
	// compute the natrual residual at x
	void NaturalResidual(const DenseVariable& x);
	// compute the penalized natural residuala t x
	void PenalizedNaturalResidual(const DenseVariable& x);
	void Copy(const DenseResidual &x);
	void Fill(double a);
	// norms and merit functions
	double Norm() const; // 2 norm
	double Merit(); // 2 norm squared
	double AbsSum(); // 1 norm

 private:
 	DenseData *data = nullptr; // access to the data object
 	static double pfb(double a, double b, double alpha);
 	static double max(double a, double b);
 	static double min(double a, double b);
 	int n,q;

};

// methods for solving linear systems + extra memory if needed
class DenseLinearSolver{
 public:
 	struct Point2D {double x; double y;};

 	DenseLinearSolver(QPsize size);
 	~DenseLinearSolver();

 	void LinkData(DenseData *data);
 	void SetAlpha(double alpha_);
 	
	bool Factor(const DenseVariable &x, const DenseVariable &xbar, double sigma);
	bool Solve(const DenseResidual &r, DenseVariable *x);

	double alpha = 0.95;
	double zero_tol = 1e-13;

	StaticMatrix K; // memory for the augmented Hessian

	StaticMatrix r1; // memory for the residuals
 	StaticMatrix r2;

 	StaticMatrix Gamma; // the FB barriers/derivatives
 	StaticMatrix mus;
 	StaticMatrix gamma;
 private:
 	DenseData *data = nullptr;

 	int n,q;

 	// computes the pfb gradient at (a,b)
 	Point2D PFBgrad(double a, double b, double sigma);

};

class DenseFeasibilityCheck{
 public:
 
 	DenseFeasibilityCheck(QPsize size);
 	~DenseFeasibilityCheck();

 	void LinkData(DenseData *data);

 	void CheckFeasibility(const DenseVariable &x, double tol);

 	bool Dual();
 	bool Primal();
  private:
  	StaticMatrix z1;
  	StaticMatrix z2;
  	StaticMatrix v1;
  	int n,q;

  	bool primal_ = true;
  	bool dual_ = true;

};


}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
