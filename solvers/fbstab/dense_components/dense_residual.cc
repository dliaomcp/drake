#define EIGEN_RUNTIME_NO_MALLOC 
#include "drake/solvers/fbstab/dense_components/dense_residual.h"

#include <cmath>
#include <Eigen/Dense>
#include <iostream>

#include "drake/solvers/fbstab/dense_components/dense_variable.h"
#include "drake/solvers/fbstab/dense_components/dense_data.h"

namespace drake {
namespace solvers {
namespace fbstab {

DenseResidual::DenseResidual(int nz, int nv){
	#ifdef EIGEN_RUNTIME_NO_MALLOC
	Eigen::internal::set_is_malloc_allowed(true);
	#endif

	nz_ = nz;
	nv_ = nv;
	z_.resize(nz_);
	v_.resize(nv_);

	#ifdef EIGEN_RUNTIME_NO_MALLOC
	Eigen::internal::set_is_malloc_allowed(false);
	#endif
}

void DenseResidual::Negate(){
	z_ *= -1.0;
	v_ *= -1.0;
}

void DenseResidual::NaturalResidual(const DenseVariable& x){
	if(data_ == nullptr){
		throw std::runtime_error("DenseResidual::NaturalResidual cannot be used unless data is linked");
	}
	// rz = H*z + f + A'*v
	// calls are arranged to avoid creating temporaries
	z_.noalias() = data_->H() * x.z();
	z_.noalias() += data_->f();
	z_.noalias() += data_->A().transpose()*x.v();

	// rv = min(y,v)
	v_.noalias() = x.y().cwiseMin(x.v());

	znorm_ = z_.norm();
	vnorm_ = v_.norm();
}

void DenseResidual::PenalizedNaturalResidual(const DenseVariable& x){
	if(data_ == nullptr){
		throw std::runtime_error("DenseResidual::PenalizedNaturalResidual cannot be used unless data is linked");
	}
	// rz = H*z + f + A'*v
	// calls are arranged to avoid creating temporaries
	z_.noalias() = data_->H() * x.z();
	z_.noalias() += data_->f();
	z_.noalias() += data_->A().transpose()*x.v();

	// rv = min(y,v) + max(0,y)*max(0,v)
	for(int i = 0; i < nv_; i++){
		v_(i) = min(x.y()(i),x.v()(i));
		v_(i) = alpha_*v()(i) + (1.0-alpha_)*max(0.0,x.y()(i))*max(0.0,x.v()(i));
	}

	znorm_ = z_.norm();
	vnorm_ = v_.norm();
}

void DenseResidual::InnerResidual(const DenseVariable& x, const DenseVariable& xbar, double sigma){
	if(data_ == nullptr){
		throw std::runtime_error("DenseResidual::InerResidual cannot be used unless data is linked");
	}
	// rz = Hz + f + A'v + sigma(z - zbar)
	// calls are arranged to avoid creating temporaries
	z_.noalias() = data_->H() * x.z();
	z_.noalias() += data_->f();
	z_.noalias() += data_->A().transpose()*x.v();
	z_.noalias() += sigma*(x.z() - xbar.z());

	// v_ = phi(ys,v), ys = y + sigma(x.v - xbar.v)
	for(int i = 0; i < nv_; i++){
		double ys = x.y()(i) + sigma*(x.v()(i) - xbar.v()(i));
		v_(i) = pfb(ys,x.v()(i),alpha_);
	}

	znorm_ = z_.norm();
	vnorm_ = v_.norm();
}

void DenseResidual::Copy(const DenseResidual &x){
	if(nz_ != x.nz_ || nv_ != x.nv_){
		throw std::runtime_error("Size mismatch in DenseResidual::Copy");
	}
	z_ = x.z_;
	v_ = x.v_;
}

void DenseResidual::Fill(double a){
	z_.fill(a);
	v_.fill(a);
}

double DenseResidual::Norm() const{
	return znorm_ + vnorm_;
}

double DenseResidual::Merit() const{
	double temp = Norm();
	return 0.5*temp*temp;
}

double DenseResidual::max(double a, double b){
	return a>b ? a : b;
}

double DenseResidual::min(double a, double b){
	return a<b ? a : b;
}

double DenseResidual::pfb(double a, double b, double alpha){
	double fb = a + b - sqrt(a*a + b*b);
	return alpha * fb + (1.0-alpha)* max(0,a)*max(0,b);
}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake

