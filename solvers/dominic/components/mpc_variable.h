#pragma once

#include "drake/solvers/dominic/linalg/static_matrix.h"
#include "drake/solvers/dominic/linalg/matrix_sequence.h"
#include "drake/solvers/dominic/components/mpc_data.h"

namespace drake {
namespace solvers {
namespace fbstab {

class MSVariable{
 public:
 	MSVariable(QPsizeMPC size);
 	~MSVariable();

 	void LinkData(MPCData *data);
 	void Fill(double a);
 	void InitConstraintMargin();
 	void axpy(const MSVariable &x, double a);
 	void Copy(const MSVariable &x);
 	void ProjectDuals();
 	double Norm();
 	double InfNorm();

 	friend std::ostream &operator<<(std::ostream& output, const MSVariable &x);

 private:
 	StaticMatrix z_; // decision variables [x0,u0,x1,u1 ... xN,uN]
 	StaticMatrix l_; // co-states [l0, ..., lN];
 	StaticMatrix v_; // ieq-duals [v0,... vN];
 	StaticMatrix y_; // ieq margin, [y0, ..., yN]

 	int N_, nx_, nu_, nc_;
 	int nz_, nl_, nv_;

 	MPCData *data_ = nullptr;
};

MSVariable::MSVariable(QPsizeMPC size){
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
}

MSVariable::~MSVariable(){
	delete[] z_.data;
	delete[] l_.data;
	delete[] v_.data;
	delete[] y_.data;
}

void MSVariable::LinkData(MPCData *data){
	data_ = data;
}

void MSVariable::Fill(double a){
	z_.fill(a);
	l_.fill(a);
	v_.fill(a);
	InitConstraintMargin();
}

void MSVariable::InitConstraintMargin(){
	// y = b-A*z
	y_.fill(0.0);
	data_->axpyb(1.0,&y_);
	data_->gemvA(z_,-1.0,1.0,&y_);
}

void MSVariable::axpy(const MSVariable &x, double a){
	z_.axpy(x.z_,a);
	l_.axpy(x.l_,a);
	v_.axpy(x.v_,a);

	// y <- y + a*(x.y - b)
	y_.axpy(x.y_,a);
	data_.axpyb(-a,&y_);
}

void MSVariable::Copy(const MSVariable &x){
	z_.copy(x.z_);
	l_.copy(x.l_);
	v_.copy(x.v_);
	y_.copy(x.y_);
	data_ = x.data_;
}

void MSVariable::ProjectDuals(){
	v_.clip(0.0,1e15);
}

double MSVariable::Norm(){
	return z_.norm() + l_.norm() + v_.norm();
}

double MSVariable::InfNorm(){
	return z_.infnorm() + l_.infnorm() + v_.infnorm();
}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake