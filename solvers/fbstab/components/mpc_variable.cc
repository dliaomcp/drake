#include "drake/solvers/fbstab/components/mpc_variable.h"

#include "drake/solvers/fbstab/linalg/static_matrix.h"
#include "drake/solvers/fbstab/linalg/matrix_sequence.h"
#include "drake/solvers/fbstab/components/mpc_data.h"


namespace drake {
namespace solvers {
namespace fbstab {


MPCVariable::MPCVariable(QPsizeMPC size){
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
	double *y_mem = new double[nv_];

	z_ = StaticMatrix(z_mem,nz_);
	l_ = StaticMatrix(l_mem,nl_);
	v_ = StaticMatrix(v_mem,nv_);
	y_ = StaticMatrix(y_mem,nv_);

	z_.fill(0.0);
	l_.fill(0.0);
	v_.fill(0.0);
	y_.fill(0.0);

	memory_allocated_ = true;
}

MPCVariable::MPCVariable(QPsizeMPC size, double *z, double *l, double *v, double *y){
	N_ = size.N;
	nx_ = size.nx;
	nu_ = size.nu;
	nc_ = size.nc;

	nz_ = (N_+1)*(nx_+nu_);
	nl_ = (N_+1)*nx_;
	nv_ = (N_+1)*nc_;

	z_ = StaticMatrix(z,nz_);
	l_ = StaticMatrix(l,nl_);
	v_ = StaticMatrix(v,nv_);
	y_ = StaticMatrix(y,nv_);

	memory_allocated_ = false;
}

MPCVariable::~MPCVariable(){
	if(memory_allocated_){
		delete[] z_.data;
		delete[] l_.data;
		delete[] v_.data;
		delete[] y_.data;
	}
}

void MPCVariable::LinkData(MPCData *data){
	data_ = data;
}

void MPCVariable::Fill(double a){
	z_.fill(a);
	l_.fill(a);
	v_.fill(a);
	InitConstraintMargin();
}

void MPCVariable::InitConstraintMargin(){
	if(data_ == nullptr)
		throw std::runtime_error("Data not linked in MPCVariable");
	// y = b-A*z
	y_.fill(0.0);
	data_->axpyb(1.0,&y_);
	data_->gemvA(z_,-1.0,1.0,&y_);
}

void MPCVariable::axpy(const MPCVariable &x, double a){
	if(data_ == nullptr)
		throw std::runtime_error("Data not linked in MPCVariable");
	z_.axpy(x.z_,a);
	l_.axpy(x.l_,a);
	v_.axpy(x.v_,a);

	// y <- y + a*(x.y - b)
	y_.axpy(x.y_,a);
	data_->axpyb(-a,&y_);
}

void MPCVariable::Copy(const MPCVariable &x){
	z_.copy(x.z_);
	l_.copy(x.l_);
	v_.copy(x.v_);
	y_.copy(x.y_);
	data_ = x.data_;
}

void MPCVariable::ProjectDuals(){
	v_.clip(0.0,1e15);
}

double MPCVariable::Norm(){
	return z_.norm() + l_.norm() + v_.norm();
}

double MPCVariable::InfNorm(){
	return z_.infnorm() + l_.infnorm() + v_.infnorm();
}

std::ostream &operator<<(std::ostream& output, const MPCVariable &x){
	std::cout << "Printing MPCVariable\n";

	std::cout << "z = [\n" << x.z_ << "]" << std::endl;
	std::cout << "l = [\n" << x.l_ << "]" << std::endl;
	std::cout << "v = [\n" << x.v_ << "]" << std::endl;

	return output;
}


}  // namespace fbstab
}  // namespace solvers
}  // namespace drake