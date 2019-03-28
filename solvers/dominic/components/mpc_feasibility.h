#pragma once

#include <cmath>

#include "drake/solvers/dominic/linalg/static_matrix.h"
#include "drake/solvers/dominic/linalg/matrix_sequence.h"
#include "drake/solvers/dominic/components/mpc_data.h"
#include "drake/solvers/dominic/components/mpc_variable.h"
#include "drake/solvers/dominic/components/mpc_residual.h"

namespace drake {
namespace solvers {
namespace fbstab {

class MPCFeasibility{
 public:
 	MPCFeasibility(QPsizeMPC size);
 	~MPCFeasibility();
 	void LinkData(MPCData *data);

 	void CheckFeasibility(const MSVariable &x, double tol);

 	bool Dual();
 	bool Primal();

  private:
  	StaticMatrix z_;
  	StaticMatrix l_;
  	StaticMatrix v_;
  	
  	bool primal_ = true;
  	bool dual_ = true;
  	int nx_, nu_, nc_, N_, nv_, nl_, nv_;
  	MPCData *data_ = nullptr;
};


MPCFeasibility::MPCFeasibility(QPsizeMPC size){
	nx_ = size.nx;
	nu_ = size.nu;
	nc_ = size.nc;
	N_ = size.N;

	nz_ = (nx_+nu_)*(N_+1);
	nl_ = nx_*(N_+1);
	nc_ = nc_*(N_+1);

	double *r1 = new double[nz_];
	double *r2 = new double[nl_];
	double *r3 = new double[nv_];

	z_ = StaticMatrix(r1,nz_);
	l_ = StaticMatrix(r2,nl_);
	v_ = StaticMatrix(r3,nv_);
}

MPCFeasibility::~MPCFeasibility(){
	delete[] z_.data;
	delete[] l_.data;
	delete[] v_.data;
}

void MPCFeasibility::LinkData(MPCData *data){
	data_ = data;
}

void MPCFeasibility::CheckFeasibility(const MSVariable &x, double tol){
	if(data_ == nullptr)
		throw std::runtime_error("Data not linked in MPCFeasibility");

	primal_ = true;
	dual_ = true;

	// check dual feasibility
	double w = x.z_.infnorm();

	// v = max(Az)
	data_->gemvA(x.z_,1.0,0.0,&v_);
	double d1 = v_.max();
	// l = norm(Gz)
	data_->gemvG(x.z_,1.0,0.0,&l_);
	double d2 = l_.infnorm();
	// z = norm(Hz)
	data_->gemvH(x.z_,1.0,0.0,&z_);
	double d3 = z_.infnorm();
	// f'*z
	z_.fill(0.0);
	data_->axpyf(1.0,&z_);
	double d4 = StaticMatrix::dot(z_,x.z_);

	if( (d1 <= 0) && (d2 <= tol*w) && (d3 <= tol*w) && (d4 < 0) && (w > 1e-14) ){
		dual_ = false;
	}

	// check primal feasibility
	double u = x.l_.infnorm() + x.v_.infnorm();

	// norm(G'*l + A'*v)
	z_.fill(0.0);
	data_->gemvAT(x.v_,1.0,1.0,&z_);
	data_->gemvGT(x.l_,1.0,1.0,&z_);
	double p1 = z_.infnorm();

	// b'*v + h'*l
	v_.fill(0.0);
	data_->axpyb(1.0,&v_);

	l_.fill(0.0);
	data_->axpyh(1.0,&l_);

	double p2 = StaticMatrix::dot(x.l_,l_) + StaticMatrix::dot(x.v_,v_);

	if( (p1 <= tol*u) && (p2 < 0) ){
		primal_ = false;
	}
}

bool MPCFeasibility::Primal(){
	return primal_;
}

bool MPCFeasibility::Dual(){
	return dual_;
}


}  // namespace fbstab
}  // namespace solvers
}  // namespace drake