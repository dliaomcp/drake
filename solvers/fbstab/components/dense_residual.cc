#include "drake/solvers/fbstab/components/dense_components.h"

#include <cmath>

#include "drake/solvers/fbstab/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {


DenseResidual::DenseResidual(DenseQPsize size){
	n_ = size.n;
	q_ = size.q;

	// allocate memory
	// TODO Eigen me
	double *r1 = new double[n_];
	double *r2 = new double[q_];

	z_ = StaticMatrix(r1,n_);
	v_ = StaticMatrix(r2,q_);
}

DenseResidual::~DenseResidual(){
	delete[] z_.data_;
	delete[] v_.data_;
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
		throw std::runtime_error("Data not liked in DenseResidual");
	}

	// rv = H*z + f + A'*v
	z_.fill(0.0);
	z_ += data_->f_;
	z_.gemv(data_->H_,x.z(),1.0,1.0); // += H*z
	z_.gemv(data_->A_,x.v(),1.0,1.0,true); // += A'*v

	// rv = min(y,v)
	for(int i = 0;i<q_;i++){
		v_(i) = min(x.y()(i),x.v()(i));
	}

	znorm_ = z_.norm();
	vnorm_ = v_.norm();
}

void DenseResidual::PenalizedNaturalResidual(const DenseVariable& x){
	if(data_ == nullptr){
		throw std::runtime_error("Data not liked in DenseResidual");
	}

	// rz = H*z + f + A'*v
	z_.fill(0.0);
	z_ += data_->f_;
	z_.gemv(data_->H_,x.z(),1.0,1.0); // += H*z
	z_.gemv(data_->A_,x.v(),1.0,1.0,true); // += A'*v

	// rv = min(y,v) + max(0,y)*max(0,v)
	for(int i = 0;i<q_;i++){
		v_(i) = min(x.y()(i),x.v()(i));
		v_(i) = alpha*v_(i) + (1.0-alpha)*max(0.0,x.y()(i))*max(0.0,x.v()(i));
	}

	znorm_ = z_.norm();
	vnorm_ = v_.norm();
}

void DenseResidual::residual(const DenseVariable& x, const DenseVariable& xbar, double sigma){
	if(data_ == nullptr){
		throw std::runtime_error("Data not liked in DenseResidual");
	}

	// z_ = Hz + f + A'v + sigma(z - zbar)
	z_.fill(0.0);
	z_ += data_->f_;
	z_.gemv(data_->H_,x.z(),1.0,1.0); // += H*z
	z_.gemv(data_->A_,x.v(),1.0,1.0,true); // += A'*v
	z_.axpy(x.z(),sigma);
	z_.axpy(xbar.z(),-1.0*sigma);

	// v_ = phi(ys,v), ys = y + sigma(x.v - xbar.v)
	for(int i = 0;i<q_;i++){
		double ys = x.y()(i) + sigma*(x.v()(i) - xbar.v()(i));
		v_(i) = pfb(ys,x.v()(i),alpha);
	}

	znorm_ = z_.norm();
	vnorm_ = v_.norm();
}

void DenseResidual::Copy(const DenseResidual &x){
	z_.copy(x.z_);
	v_.copy(x.v_);
}

void DenseResidual::Fill(double a){
	z_.fill(a);
	v_.fill(a);
}

double DenseResidual::Norm() const{
	return znorm_ + vnorm_;
}

double DenseResidual::AbsSum() const{
	return z_.asum() + v_.asum();
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

