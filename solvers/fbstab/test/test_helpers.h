#pragma once

#include "drake/solvers/fbstab/components/dense_variable.h"
#include <Eigen/Dense>

namespace drake {
namespace solvers {
namespace fbstab {
namespace test {


using VectorXd = Eigen::VectorXd;

// computes gamma and mus
void compute_pfb_derivatives(const VectorXd& y, const VectorXd& v, const VectorXd& vbar, double sigma, VectorXd* gamma, VectorXd* mus){

	int q = y.size();

	for(int i = 0; i<q;i++){
		double ys = y(i) + sigma*(v(i) - vbar(i))
	}

}

// computes the derivatives of the penalized FB function
Point2D DenseLinearSolver::PFBGradient(double a, double b){
	double y = 0;
	double x = 0;
	double r = sqrt(a*a + b*b);
	double d = 1.0/sqrt(2.0);

	if(r < zero_tolerance_){
		x = alpha_*(1.0-d);
		y = alpha_*(1.0-d);

	} else if((a > 0) && (b > 0)){
		x = alpha_ * (1.0- a/r) + (1.0-alpha_) * b;
		y = alpha_ * (1.0- b/r) + (1.0-alpha_) * a;

	} else {
		x = alpha_ * (1.0 - a/r);
		y = alpha_ * (1.0 - b/r);
	}

	Point2D output = {x, y};
	return output;
}

}  // namespace test
}  // namespace fbstab
}  // namespace solvers
}  // namespace drake