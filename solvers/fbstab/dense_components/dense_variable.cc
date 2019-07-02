#include "drake/solvers/fbstab/dense_components/dense_variable.h"

#include <cmath>
#include <Eigen/Dense>

#include "drake/solvers/fbstab/dense_components/dense_data.h"

namespace drake {
namespace solvers {
namespace fbstab {

using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;

DenseVariable::DenseVariable(int nz, int nv){
	nz_ = nz;
	nv_ = nv;

	z_ = new VectorXd(nz_);
	v_ = new VectorXd(nv_);
	y_ = new VectorXd(nv_);
	memory_allocated_ = true;
}

DenseVariable::DenseVariable(VectorXd* z, VectorXd* v, VectorXd* y){
	if(z == nullptr || v == nullptr || y == nullptr){
		throw std::runtime_error("DenseVariable::DenseVariable requires non-null pointers.");
	}
	if(y->size() != v->size()){
		throw std::runtime_error("In DenseVariable::DenseVariable: v and y input size mismatch");
	}
	nz_ = z->size();
	nv_ = v->size();
	z_ = z;
	v_ = v;
	y_ = y;
	memory_allocated_ = false;
}

DenseVariable::~DenseVariable(){
	if(memory_allocated_){
		delete z_;
		delete v_;
		delete y_;
	}
}

void DenseVariable::LinkData(const DenseData *data){
	data_ = data;
}

void DenseVariable::Fill(double a){
	if(data_ == nullptr){
		throw std::runtime_error("Cannot call DenseVariable::Fill unless data is linked");
	}
	z_->fill(a);
	v_->fill(a);
	
	// Compute y = b - A*z
	if(a == 0.0){
		*y_ = data_->b();
	} else{
		y_->noalias() = data_->b() - data_->A() * (*z_);
	}
}

void DenseVariable::InitializeConstraintMargin(){
	if(data_ == nullptr){
		throw std::runtime_error("Cannot call DenseVariable::InitializeConstraintMargin unless data is linked");
	}
	y_->noalias() = data_->b() - data_->A() * (*z_);
}

void DenseVariable::axpy(const DenseVariable &x, double a){
	if(data_ == nullptr){
		throw std::runtime_error("Cannot call DenseVariable::axpy unless data is linked");
	}
	(*z_) += a*x.z();
	(*v_) += a*x.v();
	(*y_) += a*(x.y() - data_->b());
}

void DenseVariable::Copy(const DenseVariable &x){
	if(nz_ != x.nz_ || nv_ != x.nv_){
		throw std::runtime_error("Sizes not equal in DenseVariable::Copy");
	}
	(*z_) = x.z();
	(*v_) = x.v();
	(*y_) = x.y();
	data_ = x.data_;
}

void DenseVariable::ProjectDuals(){
	*v_ = v_->cwiseMax(0);
}

double DenseVariable::Norm() const{
	return z_->norm() + v_->norm();
}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake

