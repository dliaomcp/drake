#include "drake/solvers/dominic/dense_components.h"

#include <cmath>

#include "drake/solvers/dominic/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {


// DenseData *************************************

DenseData::DenseData(double *H,double *f, 
	double *A,double *b, QPSize size){

	n = size.n;
	q = size.q;

	// create static matrices over the inputs
	this->H = StaticMatrix(H,n,n);
	this->f = StaticMatrix(f,n);
	this->A = StaticMatrix(A,q,n);
	this->b = StaticMatrix(b,q);

}

// DenseVariable *************************************

DenseVariable::DenseVariable(QPsize size){
	n = size.n;
	q = size.q;

	double *z_mem = new double[n];
	double *v_mem = new double[q];
	double *y_mem = new double[q];

	z = StaticMatrix(z_mem,n);
	v = StaticMatrix(v_mem,q);
	y = StaticMatrix(y_mem,q);
	z.fill(0.0);
	v.fill(0.0);
	y.fill(0.0);
}

DenseVariable::~DenseVariable(){
	delete[] z.data;
	delete[] v.data;
	delete[] y.data;
}

void DenseVariable::LinkData(DenseData *data){
	this->data = data;
}

void DenseVariable::Fill(double a){
	z.fill(a);
	v.fill(a);
	
	if(data != nullptr){
		// y = b - A*z
		y.copy(data->b);
		if(a~= 0.0){
			y.gemv(data->A,z,-1.0,1.0);
		}

	} else{
		throw std::runtime_error("Cannot call DenseVariable::Fill unless data is linked");
	}
}

void DenseVariable::InitConstraintMargin(){
	if(data!= nullptr){
		y.copy(data->b);
		y.gemv(data->A,z,-1.0,1.0);
	} else{
		throw std::runtime_error("Cannot initialize constraint margin until data is linked");
	}
}

void DenseVariable::axpy(const DenseVariable &x, double a){
	z.axpy(x.z,a);
	v.axpy(x.v,a);
	// y <- y + a*(x.y - b)
	y.axpy(x.y,a);

	if(data != nullptr)
		y.axpy(data->b,-a);
	else
		throw std::runtime_error("Cannot call DenseVariable::axpy unless data is linked");
}

void DenseVariable::Copy(const DenseVariable &x){
	z.copy(x.z);
	v.copy(x.v);
	y.copy(x.y);
	data = x.data;
}

void DenseVariable::ProjectDuals(){
	v.clip(0.0,1e15);
}

// DenseResidual *************************************

DenseResidual::DenseResidual(QPsize size){
	n = size.n;
	q = size.q;

	// allocate memory
	double *r1 = new double[n];
	double *r2 = new double[q];

	rz = StaticMatrix(r1,n);
	rv = StaticMatrix(r2,q);
}

DenseResidual::~DenseResidual(){
	delete[] rz.data;
	delete[] rv.data;
}

void DenseResidual::LinkData(DenseData *data){
	this->data = data;
}

void DenseResidual::Negate(){
	rz *= -1.0;
	rv *= -1.0;
}

void DenseResidual::NaturalResidual(const DenseVariable& x){
	// rz = H*z + f + A'*v
	rz.fill(0.0);
	rz.axpy(data->f,0.0); // += f 
	rz.gemv(data->H,x.z,1.0,1.0); // += H*z
	rz.gemv(data->A,x.v,1.0,1.0,true); // += A'*v

	// rv = min(y,v)
	for(int i = 0;i<q;i++){
		rv(i) = StaticMatrix::min(x.y(i),x.v(i));
	}
}

void DenseResidual::PenalizedNaturalResidual(const DenseVariable& x){

}

void DenseResidual::FBresidual(const DenseVariable& x, 
		const DenseVariable& xbar, double sigma){

}

double DenseResidual::Norm(){
	return rz.norm() + rv.norm();
}

double DenseResidual::Merit(){
	double temp = this->Norm();
	return 0.5*temp*temp;
}


// DenseLinearSolver *************************************


}  // namespace fbstab
}  // namespace solvers
}  // namespace drake

