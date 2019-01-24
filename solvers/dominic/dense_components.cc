#include "drake/solvers/dominic/dense_components.h"

#include <cmath>

#include "drake/solvers/dominic/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace dominic {


// DenseData *************************************

DenseData::DenseData(double *H,double *f, 
	double *A,double *b, QPsize size){

	n = size.n;
	q = size.q;

	// TODO: add check for null

	// create static matrices over the inputs
	this->H.map(H,n,n);
	this->f.map(f,n);
	this->A.map(A,q,n);
	this->b.map(b,q);

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

	memory_allocated = true;
}

DenseVariable::DenseVariable(QPsize size, double *z_mem, 
	double *v_mem, double y_mem){
	n = size.n;
	q = size.q;

	z = StaticMatrix(z_mem,n);
	v = StaticMatrix(v_mem,q);
	y = StaticMatrix(y_mem,q);

	memory_allocated = false;
}

DenseVariable::~DenseVariable(){
	if(memory_allocated){
		delete[] z.data;
		delete[] v.data;
		delete[] y.data;
	}
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

// // DenseResidual *************************************

// // TODO: throw an error if anything is done before
// // linking a data object?
// DenseResidual::DenseResidual(QPsize size){
// 	n = size.n;
// 	q = size.q;

// 	// allocate memory
// 	double *r1 = new double[n];
// 	double *r2 = new double[q];

// 	rz = StaticMatrix(r1,n);
// 	rv = StaticMatrix(r2,q);
// }

// DenseResidual::~DenseResidual(){
// 	delete[] rz.data;
// 	delete[] rv.data;
// }

// void DenseResidual::LinkData(DenseData *data){
// 	this->data = data;
// }

// void DenseResidual::Negate(){
// 	rz *= -1.0;
// 	rv *= -1.0;
// }

// void DenseResidual::NaturalResidual(const DenseVariable& x){
// 	// rz = H*z + f + A'*v
// 	rz.fill(0.0);
// 	rz += data->f;
// 	rz.gemv(data->H,x.z,1.0,1.0); // += H*z
// 	rz.gemv(data->A,x.v,1.0,1.0,true); // += A'*v

// 	// rv = min(y,v)
// 	for(int i = 0;i<q;i++){
// 		rv(i) = StaticMatrix::min(x.y(i),x.v(i));
// 	}
// }

// void DenseResidual::PenalizedNaturalResidual(const DenseVariable& x){
// 	// rz = H*z + f + A'*v
// 	rz.fill(0.0);
// 	rz += data->f;
// 	rz.gemv(data->H,x.z,1.0,1.0); // += H*z
// 	rz.gemv(data->A,x.v,1.0,1.0,true); // += A'*v

// 	// rv = min(y,v)
// 	for(int i = 0;i<q;i++){
// 		rv(i) = StaticMatrix::min(x.y(i),x.v(i));
// 		rv(i) = alpha*rv(i) + 
// 			(1.0-alpha)*max(0.0,x.y(i))*max(0.0,x.v(i));
// 	}

// }

// void DenseResidual::FBresidual(const DenseVariable& x, 
// 		const DenseVariable& xbar, double sigma){

// 	// rz = Hz + f + A'v + sigma(z - zbar)
// 	rz.fill(0.0);
// 	rz += data->f;
// 	rz.gemv(data->H,x.z,1.0,1.0); // += H*z
// 	rz.gemv(data->A,x.v,1.0,1.0,true); // += A'*v
// 	rz.axpy(x.z,sigma);
// 	rz.axpy(xbar.z,-sigma);

// 	// rv = phi(ys,v), ys = y + sigma(x.v - xbar.v)
// 	for(int i = 0;i<q;i++){
// 		double ys = x.y(i) + sigma*(x.v(i) - xbar.v(i));
// 		rv(i) = pfb(ys,x.v(i));
// 	}

// }

// double DenseResidual::Norm(){
// 	return rz.norm() + rv.norm();
// }

// double DenseResidual::Merit(){
// 	double temp = this->Norm();
// 	return 0.5*temp*temp;
// }

// double DenseResidual::max(double a, double b){
// 	return a>b ? a : b;
// }

// double DenseResidual::min(double a, double b){
// 	return a<b ? a : b;
// }

// double DenseResidual::pfb(double a, double b, double alpha){
// 	double fb = a + b - sqrt(a*a + b*b);
// 	return alpha * fb + (1.0-alpha)* max(0,a)*max(0,b);
// }

// // DenseLinearSolver *************************************

// DenseLinearSolver::DenseLinearSolver(QPsize size){
// 	n = size.n;
// 	q = size.q;

// 	// fb derivatives
// 	double *a1 = new double[q];
// 	double *a2 = new double[q];
// 	double *a3 = new double[q];
// 	gamma = StaticMatrix(a1,q);
// 	mu = StaticMatrix(a2,q);
// 	Gamma = StaticMatrix(a3,q);

// 	// workspace residuals
// 	double *b1 = new double[n];
// 	double *b2 = new double[q];
// 	r1 = StaticMatrix(b1,n);
// 	r2 = StaticMatrix(b2,q);

// 	// workspace hessian
// 	double *c = new double[n*n];
// 	K = StaticMatrix(c,n,n);
// }

// DenseLinearSolver::~DenseLinearSolver(){
// 	delete[] gamma.data;
// 	delete[] mu.data;
// 	delete[] Gamma.data;

// 	delete[] r1.data;
// 	delete[] r2.data;

// 	delete[] K.data;
// }

// void DenseLinearSolver::LinkData(const DenseData *data){
// 	this->data = data;
// }

// bool DenseLinearSolver::Factor(const DenseVariable &x, double sigma){

// 	// compute K = H + sigma I + A'*Gamma A
// 	K.copy(data->H);
// 	for(int i = 0;i<K.rows();i++){
// 		K(i,i) += sigma;
// 	}

// 	// K <- K + A'*diag(Gamma(x))*A
// 	Point2D tmp;
// 	for(int i = 0;i<q;i++){
// 		tmp = PFBgrad(x.y(i),x.v(i),sigma);
// 		gamma(i) = tmp.x;
// 		mus(i) = tmp.y + sigma*tmp.x;
// 		Gamma(i) = gamma(i)/mus(i);
// 	}
// 	K.gram(data->A,Gamma);

// 	// K = LL' in place
// 	int chol_flag = K.llt();

// 	if(chol_flag == 0)
// 		return true;
// 	else
// 		return false;
// }

// bool DenseLinearSolver::Solve(const DenseResidual &r, double sigma, DenseVariable *x){
// 	// solve the system
// 	// K z = rz - A'*diag(1/mus)*rv
// 	// diag(mus) v = rv + diag(gamma)*A*z

// 	for(int i = 0;i<r1.rows();i++){
// 		mus(i) = 1.0/mus(i);
// 	}

// 	r1.copy(r.rz);
// 	r2.copy(r.rv);
// 	// r2 = rv./mus
// 	r2.RowScale(mus);
// 	// r1 = -1.0*A'*r2 + 1.0*r1
// 	r1.gemv(data->A,r2,-1.0,1.0,true); 

// 	// solve K z = r1
// 	r1.CholSolve(K);
// 	(x->z).copy(r1);

// 	// r2 = rv + diag(gamma)*A*z
// 	r2.gemv(data->A,x->z,1.0);
// 	r2.RowScale(gamma);
// 	r2.axpy(r.rv,1.0);

// 	// v = diag(1/mus)*r2
// 	r2.RowScale(mus);
// 	(x->v).copy(r2);

// 	// y = b - Az + sigma v
// 	r2.copy(data->b);
// 	r2.gemv(data->A,x->z,-1.0);
// 	r2.axpy(x->v,sigma);

// 	return true;
// }


// Point2D DenseLinearSolver::PFBgrad(double a, double b, double sigma){
// 	double mu = 0;
// 	double gamma = 0;
// 	double r = sqrt(a*a + b*b);
// 	double d = sqrt(2.0);

// 	if(r < zero_tol){
// 		gamma = alpha*(1.0-d);
// 		mu = alpha*(1.0-d);

// 	} else if((a > 0) && (b > 0)){
// 		gamma = alpha * (1.0- a/r) + (1.0-alpha) * b;
// 		mu = alpha * (1.0- b/r) + (1.0-alpha) * a;

// 	} else {
// 		gamma = alpha * (1.0 - a/r);
// 		mu = alpha * (1.0 - b/r);
// 	}

// 	Point2D out = {gamma, mu};
// 	return out;
// }


}  // namespace fbstab
}  // namespace solvers
}  // namespace drake

