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

class MSVariable{
 public:
 	MSVariable(QPsizeMPC size);
 	~MSVariable();

 	void LinkData(MPCData *data);
 	void Fill(double a);
 	void InitConstraintMargin();
 	void axpy(const MSVariable &x, double a);
 	void Copy(const MSVariable &x);
 	void ProjectDuals();
 	double Norm();
 	double InfNorm();

 	friend std::ostream &operator<<(std::ostream& output, const MSVariable &x);

 private:
 	StaticMatrix x_; // states [x0,x1 ... xN]
 	StaticMatrix u_; // controls [u0,u1, ... uN]
 	StaticMatrix l_; // co-states [l0, ..., lN];
 	StaticMatrix v_; // ieq-duals [v0,... vN];
 	StaticMatrix y_; // ieq margin, [y0, ..., yN]

 	int N_, nx_, nu_, nc_;
 	int nz_, nl_, nv_;

 	MPCData *data_ = nullptr;
};

class MSResidual{
 public:
 	double z_norm;
 	double v_norm;

 	MSResidual(QPsizeMPC size);
 	~MSResidual();
 	void LinkData(MPCData *data);
 	void Negate();
 	void FBresidual(const MSVariable &x, const MSVariable &xbar, double sigma);
 	void NaturalResidual(const MSVariable &x);
 	void PenalizedNaturalResidual(const MSVariable &x);
 	void Copy(const MSResidual &x);
 	void Fill(double a);
 	double Norm() const;
 	double Merit() const;
 	double AbsSum() const;

 private:
 	StaticMatrix z_;
 	StaticMatrix l_;
 	StaticMatrix v_;

 	int N_, nx_, nu_, nc_;
 	int nz_, nl_, nv_;

 	double alpha_ = 0.95;
 	MPCData *data_ = nullptr;

 	static double pfb(double a, double b, double alpha);
 	static double max(double a, double b);
 	static double min(double a, double b);
};

class RicattiLinearSolver{
 public:
 	struct Point2D {double x; double y;};

 	RicattiLinearSolver(QPsizeMPC size);
 	~RicattiLinearSolver();

 	void LinkData(MPCData *data);
 	bool Factor(const MSVariable &x, const MSVariable &xbar, double sigma);
 	bool Solve(const MSResidual &r, MSVariable *dx);

 private:
 	int N_, nx_, nu_, nc_;
 	int nz_, nl_, nv_;

 	double alpha_ = 0.95;
 	MPCData *data_ = nullptr;
 	double zero_tol_ = 1e-13;

 	MatrixSequence Q_, R_, S_;
 	MatrixSequence P_;
 	MatrixSequence Sig_;
 	MatrixSequence M_;
 	MatrixSequence L_;
 	MatrixSequence SM_;
 	MatrixSequence AM_;

 	MatrixSequence h_;
 	MatrixSequence theta_;

 	Point2D PFBgrad(double a, double b, double sigma);

};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
