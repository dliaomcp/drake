#pragma once

#include "drake/solvers/fbstab/linalg/static_matrix.h"


// contains the gereral dense qp specific classes and structures

namespace drake {
namespace solvers {
namespace fbstab {


class DenseResidual{
 public:
	// methods *************************************
	DenseResidual(DenseQPsize size);
	~DenseResidual();

	void LinkData(DenseData *data);
	void Negate(); // y <- -1*y

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
	double AbsSum() const; // 1 norm

	void SetAlpha(double alpha) {alpha_ = alpha};
	double z_norm() const { return znorm_ };
	double v_norm() const { return vnorm_ };
	double l_norm() const { return lnorm_ };


 private:
 	DenseData *data_ = nullptr; // access to the data object
 	int n_,q_;
 	StaticMatrix z_; // stationarity residual
	StaticMatrix v_; // complimentarity residual
	double alpha_ = 0.95; 
	double znorm_ = 0.0;
	double vnorm_ = 0.0;
	double lnorm_ = 0.0;

	static double pfb(double a, double b, double alpha);
 	static double max(double a, double b);
 	static double min(double a, double b);
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
