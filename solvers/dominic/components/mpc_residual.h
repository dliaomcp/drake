#pragma once

#include <cmath>

#include "drake/solvers/dominic/linalg/static_matrix.h"
#include "drake/solvers/dominic/linalg/matrix_sequence.h"
#include "drake/solvers/dominic/components/mpc_data.h"
#include "drake/solvers/dominic/components/mpc_residual.h"

namespace drake {
namespace solvers {
namespace fbstab {

class MSResidual{
 public:

 	MSResidual(QPsizeMPC size);
 	~MSResidual();
 	void LinkData(MPCData *data);

 	void Fill(double a);
 	void Copy(const MSResidual &x);
 	void Negate();
 	double Norm() const;
 	double Merit() const;
 	double AbsSum() const;

 	void FBresidual(const MSVariable &x, const MSVariable &xbar, double sigma);
 	void NaturalResidual(const MSVariable &x);
 	void PenalizedNaturalResidual(const MSVariable &x);
 	
 	StaticMatrix z_;
 	StaticMatrix l_;
 	StaticMatrix v_;

 	double z_norm = 0.0;
 	double v_norm = 0.0;
 	double l_norm = 0.0;

 private:
 	int N_, nx_, nu_, nc_;
 	int nz_, nl_, nv_;

 	double alpha_ = 0.95;
 	MPCData *data_ = nullptr;

 	static double pfb(double a, double b, double alpha);
 	static double max(double a, double b);
 	static double min(double a, double b);

 	bool memory_allocated_ = false;
};

MSResidual::MSResidual(QPsizeMPC size){
	N_ = size.N;
	nx_ = size.nx;
	nu_ = size.nu;
	nc_ = size.nc;

	nz_ = (N_+1)*(nx_+nu_);
	nl_ = (N_+1)*nx_;
	nv_ = (N_+1)*nc_;

	double *z_mem = new double[nz_];
	double *l_mem = new double[nl_];
	double *v_mem = new double[nv_];

	z_ = StaticMatrix(z_mem,nz_);
	l_ = StaticMatrix(l_mem,nl_);
	v_ = StaticMatrix(v_mem,nv_);

	z_.fill(0.0);
	l_.fill(0.0);
	v_.fill(0.0);

	memory_allocated_ = true;
}

MSResidual::~MSResidual(){
	if(memory_allocated_){
		delete[] z_.data;
		delete[] l_.data;
		delete[] v_.data;
	}
}

void MSResidual::LinkData(MPCData *data){
	data_ = data;
}

void MSResidual::Fill(double a){
	z_.fill(a);
	l_.fill(a);
	v_.fill(a);
}

void MSResidual::Copy(const MSResidual &x){
	z_.copy(x.z_);
	l_.copy(x.l_);
	v_.copy(x.v_);

	z_norm = x.z_norm;
	v_norm = x.v_norm;
	l_norm = x.l_norm;
}

void MSResidual::Negate(){
	z_ *= -1;
	l_ *= -1;
	v_ *= -1;
}

double MSResidual::Norm() const{
	return z_norm + l_norm + v_norm;
}

double MSResidual::Merit() const{
	double temp = this->Norm();
	return 0.5*temp*temp;
}

double MSResidual::AbsSum() const{
	return z_.asum() + l_.asum() + v_.asum();
}

void MSResidual::FBresidual(const MSVariable &x, const MSVariable &xbar, double sigma){
	if(data_ == nullptr)
		throw std::runtime_error("Data not linked in MSResidual");

	// r.z = H*z + f + G'*l + A'*v + sigma*(z-zbar)
	z_.fill(0.0);
	data_->axpyf(1.0,&z_);
	data_->gemvH(x.z_,1.0,1.0,&z_);
	data_->gemvGT(x.l_,1.0,1.0,&z_);
	data_->gemvAT(x.v_,1.0,1.0,&z_);
	z_.axpy(x.z_,sigma);
	z_.axpy(xbar.z_,-sigma);

	// r.l = h - G*z + sigma(l - lbar)
	l_.fill(0.0);
	data_->axpyh(1.0,&l_);
	data_->gemvG(x.z_,-1.0,1.0,&l_);
	l_.axpy(x.l_,sigma);
	l_.axpy(xbar.l_,-sigma);

	// rv = phi(y + sigma*(v-vbar),v)
	for(int i =0;i<nv_;i++){
		double ys = x.y_(i) + sigma*(x.v_(i)-xbar.v_(i));
		v_(i) = pfb(ys,x.v_(i),alpha_);
	}
	z_norm = z_.norm();
	l_norm = l_.norm();
	v_norm = v_.norm();
}

void MSResidual::NaturalResidual(const MSVariable &x){
	if(data_ == nullptr)
		throw std::runtime_error("Data not linked in MSResidual");

	// r.z = H*z + f + G'*l + A'*v 
	z_.fill(0.0);
	data_->axpyf(1.0,&z_);
	data_->gemvH(x.z_,1.0,1.0,&z_);
	data_->gemvGT(x.l_,1.0,1.0,&z_);
	data_->gemvAT(x.v_,1.0,1.0,&z_);

	// r.l = h - G*z + sigma(l - lbar)
	l_.fill(0.0);
	data_->axpyh(1.0,&l_);
	data_->gemvG(x.z_,-1.0,1.0,&l_);

	// rv = min(y,v)
	for(int i =0;i<nv_;i++){
		v_(i) = min(x.y_(i),x.v_(i));
	}
	z_norm = z_.norm();
	l_norm = l_.norm();
	v_norm = v_.norm();
}

void MSResidual::PenalizedNaturalResidual(const MSVariable &x){
	if(data_ == nullptr)
		throw std::runtime_error("Data not linked in MSResidual");

	// r.z = H*z + f + G'*l + A'*v 
	z_.fill(0.0);
	data_->axpyf(1.0,&z_);
	data_->gemvH(x.z_,1.0,1.0,&z_);
	data_->gemvGT(x.l_,1.0,1.0,&z_);
	data_->gemvAT(x.v_,1.0,1.0,&z_);

	// r.l = h - G*z + sigma(l - lbar)
	l_.fill(0.0);
	data_->axpyh(1.0,&l_);
	data_->gemvG(x.z_,-1.0,1.0,&l_);

	// rv = alpha*min(y,v) + (1-alpha)*max(0,v)*max(0,y)
	for(int i=0;i<nv_;i++){
		double nr = min(x.y_(i),x.v_(i));
		v_(i) = alpha_*nr + (1-alpha_)*max(0.0,x.y_(i))*max(0,x.v_(i));
	}
	z_norm = z_.norm();
	l_norm = l_.norm();
	v_norm = v_.norm();
}

double MSResidual::pfb(double a, double b, double alpha){
	double fb = a + b - sqrt(a*a + b*b);
	return alpha * fb + (1.0-alpha)* max(0,a)*max(0,b);
}

double MSResidual::max(double a, double b){
	return a>b ? a : b;
}

double MSResidual::min(double a, double b){
	return a<b ? a : b;
}


}  // namespace fbstab
}  // namespace solvers
}  // namespace drake