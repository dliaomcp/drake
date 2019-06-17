#include "drake/solvers/fbstab/components/dense_variable.h"

#include <cmath>
#include <Eigen/Dense>

#include "drake/solvers/fbstab/components/dense_data.h"
#include "drake/solvers/fbstab/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {

using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;

DenseVariable::DenseVariable(DenseQPsize size){
	n_ = size.n;
	q_ = size.q;

	z_.resize(n_);
	v_.resize(q_);
	y_.resize(q_);
}

void DenseVariable::LinkData(DenseData *data){
	data_ = data;
}

void DenseVariable::Fill(double a){
	if(data_ == nullptr){
		throw std::runtime_error("Cannot call DenseVariable::Fill unless data is linked");
	}
	z_.fill(a);
	v_.fill(a);
	
	// Compute y = b - A*z
	if(a == 0.0){
		y_ = data_->b_;
	} else{
		y_.noalias() = data_->b_ - data_->A_ * z_;
	}
}

void DenseVariable::InitializeConstraintMargin(){
	if(data_ == nullptr){
		throw std::runtime_error("Cannot call DenseVariable::InitializeConstraintMargin unless data is linked");
	}
	y_.noalias() = data_->b_ - data_->A_ * z_;
}

void DenseVariable::axpy(const DenseVariable &x, double a){
	if(data_ == nullptr){
		throw std::runtime_error("Cannot call DenseVariable::axpy unless data is linked");
	}
	z_ += a*x.z_;
	v_ += a*x.v_;
	y_ += a*(x.y_ - data_->b_);
}

void DenseVariable::Copy(const DenseVariable &x){
	if(n_ != x.n_ || q_ != x.q_){
		throw std::runtime_error("Sizes not equal in DenseVariable::Copy");
	}
	z_ = x.z_;
	v_ = x.v_;
	y_ = x.y_;
	data_ = x.data_;
}

void DenseVariable::ProjectDuals(){
	v_ = v_.cwiseMax(0);
}

double DenseVariable::Norm() const{
	return z_.norm() + v_.norm();
}

std::ostream &operator<<(std::ostream& output, const DenseVariable &x){
	std::cout << "Printing DenseVariable\n"; 

	std::cout << x.z_ << std::endl;
	std::cout << x.v_ << std::endl;

	return output;
}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake

