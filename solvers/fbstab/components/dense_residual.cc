// #define EIGEN_RUNTIME_NO_MALLOC 
#include "drake/solvers/fbstab/components/dense_residual.h"

#include <cmath>
#include <Eigen/Dense>

#include "drake/solvers/fbstab/linalg/static_matrix.h"
#include "drake/solvers/fbstab/components/dense_variable.h"
#include "drake/solvers/fbstab/components/dense_data.h"

namespace drake {
namespace solvers {
namespace fbstab {


DenseResidual::DenseResidual(int n, int q){
	#ifdef EIGEN_RUNTIME_NO_MALLOC
	Eigen::internal::set_is_malloc_allowed(true);
	#endif

	n_ = n;
	q_ = q;
	z_.resize(n_);
	v_.resize(q_);

	#ifdef EIGEN_RUNTIME_NO_MALLOC
	Eigen::internal::set_is_malloc_allowed(false);
	#endif
}

void DenseResidual::LinkData(DenseData *data){
	data_ = data;
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
	z_.noalias() = data_->H_*x.z() + data_->f_ + data_->A_.transpose()*x.v();

	// rv = min(y,v)
	v_ = x.y().cwiseMin(x.v());

	znorm_ = z_.norm();
	vnorm_ = v_.norm();
}

void DenseResidual::PenalizedNaturalResidual(const DenseVariable& x){
	if(data_ == nullptr){
		throw std::runtime_error("DenseResidual::PenalizedNaturalResidual cannot be used unless data is linked");
	}
	// rz = H*z + f + A'*v
	z_.noalias() = data_->H_*x.z() + data_->f_ + data_->A_.transpose()*x.v();

	// rv = min(y,v) + max(0,y)*max(0,v)
	for(int i = 0; i < q_; i++){
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
	z_.noalias() = data_->H_*x.z() + data_->f_ + data_->A_.transpose()*x.v();
	z_.noalias() += sigma*(x.z() - xbar.z());

	// v_ = phi(ys,v), ys = y + sigma(x.v - xbar.v)
	for(int i = 0; i < q_; i++){
		double ys = x.y()(i) + sigma*(x.v()(i) - xbar.v()(i));
		v_(i) = pfb(ys,x.v()(i),alpha_);
	}

	znorm_ = z_.norm();
	vnorm_ = v_.norm();
}

void DenseResidual::Copy(const DenseResidual &x){
	if(n_ != x.n_ || q_ != x.q_){
		throw std::runtime_error("Sizes not equal in DenseResidual::Copy");
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

