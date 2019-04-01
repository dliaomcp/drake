#pragma once

#include "drake/solvers/dominic/linalg/static_matrix.h"
#include "drake/solvers/dominic/linalg/matrix_sequence.h"

namespace drake {
namespace solvers {
namespace fbstab {


struct QPsizeMPC {
	int N;
	int nx;
	int nu;
	int nc;
};

class MPCData{
 public:
 	MPCData(double **Q, double **R, double **S, double **q, 
 			double **r, double **A, double **B, double **c,
 			double **E, double **L, double **d, double *x0, 
 			QPsizeMPC size);

 	// y <- a*T*x + b*y for T = H,A,G,A',G'
 	void gemvH(const StaticMatrix &x, double a, double b, StaticMatrix *y); 
 	void gemvA(const StaticMatrix &x, double a, double b, StaticMatrix *y); 
 	void gemvG(const StaticMatrix &x, double a, double b, StaticMatrix *y); 
 	void gemvAT(const StaticMatrix &x, double a, double b, StaticMatrix *y); 
	void gemvGT(const StaticMatrix &x, double a, double b, StaticMatrix *y);

	// y <- a*x + y for x = f,h,b
	void axpyf(double a, StaticMatrix *y);
	void axpyh(double a, StaticMatrix *y);
	void axpyb(double a, StaticMatrix *y);

 private:
 	int N_;
 	int nx_;
 	int nu_;
 	int nc_;

 	int nz_;
 	int nl_;
 	int nv_;

 	MatrixSequence Q_,R_,S_,q_,r_;
 	MatrixSequence A_,B_,c_;
 	MatrixSequence E_,L_,d_;
 	StaticMatrix x0_;

 	friend class MSVariable;
 	friend class MSResidual;
 	friend class RicattiLinearSolver;
 	friend class MSFeasibilityChecker;
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
