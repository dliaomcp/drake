#include "drake/solvers/fbstab/components/dense_variable.h"

#include <cmath>

#include "drake/solvers/fbstab/components/dense_data.h"
#include "drake/solvers/fbstab/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {

DenseVariable::DenseVariable(DenseQPsize size){
	n_ = size.n;
	q_ = size.q;

	// replace with Eigen
	double *z_mem = new double[n_];
	double *v_mem = new double[q_];
	double *y_mem = new double[q_];

	z_ = StaticMatrix(z_mem,n_);
	v_ = StaticMatrix(v_mem,q_);
	y_ = StaticMatrix(y_mem,q_ß);
	z_.fill(0.0);
	v_.fill(0.0);
	y_.fill(0.0);

	memory_allocated_ = true;
}

DenseVariable::~DenseVariable(){
	if(memory_allocated_){
		delete[] z_.data;
		delete[] v_.data;
		delete[] y_.data;
	}
}

void DenseVariable::LinkData(DenseData *data){
	data_ = data;
}

void DenseVariable::Fill(double a){
	z.fill(a);
	v.fill(a);
	
	if(data != nullptr){
		// y = b - A*z
		y_.copy(data_->b_);
		if(a != 0.0){
			y_.gemv(data->A_,z_,-1.0,1.0);
		}

	} else{
		throw std::runtime_error("Cannot call DenseVariable::Fill unless data is linked");
	}
}

void DenseVariable::InitializeConstraintMargin(){
	if(data!= nullptr){
		y_.copy(data->b_);
		y_.gemv(data->A_,z_,-1.0,1.0);
	} else{
		throw std::runtime_error("Cannot call DenseVariable::InitializeConstraintMargin unless data is linked");
	}
}

void DenseVariable::ScaledAccumulate(const DenseVariable &x, double a){
	z_.axpy(x.z_,a);
	v_.axpy(x.v_,a);

	// y <- y + a*(x.y - b)
	y_.axpy(x.y_,a);
	if(data != nullptr)
		y_.axpy(data->b_,-a);
	else
		throw std::runtime_error("Cannot call DenseVariable::ScaledAccumulate unless data is linked");
}

void DenseVariable::Copy(const DenseVariable &x){
	z_.copy(x.z_);
	v_.copy(x.v_);
	y_.copy(x.y_);
	data_ = x.data_;
}

void DenseVariable::ProjectDuals(){
	v_.clip(0.0,1e15);
}

double DenseVariable::Norm() const{
	return z_.norm() + v_.norm();
}

double DenseVariable::InfNorm() const{
	return z_.infnorm() + v_.infnorm();
}

std::ostream &operator<<(std::ostream& output, const DenseVariable &x){
	std::cout << "Printing DenseVariable\n"; 

	std::cout << "z = [\n" << x.z_ << "]" << std::endl;
	std::cout << "v = [\n" << x.v_ << "]" << std::endl;

	return output;
}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake

