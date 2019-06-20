#pragma once

#include <Eigen/Dense>

#include "drake/solvers/fbstab/components/dense_variable.h"
#include "drake/solvers/fbstab/components/dense_data.h"

namespace drake {
namespace solvers {
namespace fbstab {

class DenseResidual{
 public:
	DenseResidual(int nz, int nv);

	void LinkData(DenseData *data);

	// y <- -1*y
	void Negate(); 

	// compute the residual for the proximal subproblem at (x,xbar,sigma)
	void InnerResidual(const DenseVariable& x, 
		const DenseVariable& xbar, double sigma);

	// compute the natural residual at x
	void NaturalResidual(const DenseVariable& x);

	// compute the penalized natural residual at x
	void PenalizedNaturalResidual(const DenseVariable& x);

	void Copy(const DenseResidual &x);
	void Fill(double a);
	double Norm() const; // 2 norm
	double Merit() const; // 2 norm squared

	void SetAlpha(double alpha) {alpha_ = alpha;}

	double z_norm() const { return znorm_; }
	double v_norm() const { return vnorm_; }
	double l_norm() const { return lnorm_; }

	Eigen::VectorXd& z(){ return z_; };
	Eigen::VectorXd& v(){ return v_; };

 private:
 	DenseData *data_ = nullptr; // access to the data object
 	int nz_ = 0;
 	int nv_ = 0;
 	Eigen::VectorXd z_; // stationarity residual
	Eigen::VectorXd v_; // complimentarity residual
	double alpha_ = 0.95; 
	double znorm_ = 0.0;
	double vnorm_ = 0.0;
	double lnorm_ = 0.0;

	static double pfb(double a, double b, double alpha);
 	static double max(double a, double b);
 	static double min(double a, double b);

 	friend class DenseLinearSolver;
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
