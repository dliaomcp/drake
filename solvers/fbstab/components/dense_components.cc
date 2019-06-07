#include "drake/solvers/dominic/components/dense_components.h"

#include <cmath>

#include "drake/solvers/dominic/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {


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
	double *v_mem, double *y_mem){
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

void DenseVariable::LinkData(DenseData *data_){
	this->data = data_;
}

void DenseVariable::Fill(double a){
	z.fill(a);
	v.fill(a);
	
	if(data != nullptr){
		// y = b - A*z
		y.copy(data->b);
		if(a != 0.0){
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

double DenseVariable::Norm(){
	return z.norm() + v.norm();
}

double DenseVariable::InfNorm(){
	return z.infnorm() + v.infnorm();
}

std::ostream &operator<<(std::ostream& output, const DenseVariable &x){
	std::cout << "Printing DenseVariable\n"; 

	std::cout << "z = [\n" << x.z << "]" << std::endl;
	std::cout << "v = [\n" << x.v << "]" << std::endl;

	return output;
}

// // DenseResidual *************************************

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

void DenseResidual::LinkData(DenseData *data_){
	this->data = data_;
}

void DenseResidual::SetAlpha(double alpha_){
	alpha = alpha_;
}

void DenseResidual::Negate(){
	rz *= -1.0;
	rv *= -1.0;
}

void DenseResidual::NaturalResidual(const DenseVariable& x){
	if(data == nullptr)
		throw std::runtime_error("Data not liked in DenseResidual");
	// rz = H*z + f + A'*v
	rz.fill(0.0);
	rz += data->f;
	rz.gemv(data->H,x.z,1.0,1.0); // += H*z
	rz.gemv(data->A,x.v,1.0,1.0,true); // += A'*v

	// rv = min(y,v)
	for(int i = 0;i<q;i++){
		rv(i) = StaticMatrix::min(x.y(i),x.v(i));
	}

	z_norm = rz.norm();
	v_norm = rv.norm();
}

void DenseResidual::PenalizedNaturalResidual(const DenseVariable& x){
	if(data == nullptr)
		throw std::runtime_error("Data not liked in DenseResidual");
	// rz = H*z + f + A'*v
	rz.fill(0.0);
	rz += data->f;
	rz.gemv(data->H,x.z,1.0,1.0); // += H*z
	rz.gemv(data->A,x.v,1.0,1.0,true); // += A'*v

	// rv = min(y,v)
	for(int i = 0;i<q;i++){
		rv(i) = StaticMatrix::min(x.y(i),x.v(i));
		rv(i) = alpha*rv(i) + 
			(1.0-alpha)*max(0.0,x.y(i))*max(0.0,x.v(i));
	}

	z_norm = rz.norm();
	v_norm = rv.norm();
}

void DenseResidual::FBresidual(const DenseVariable& x, 
		const DenseVariable& xbar, double sigma){
	if(data == nullptr)
		throw std::runtime_error("Data not liked in DenseResidual");
	// rz = Hz + f + A'v + sigma(z - zbar)
	rz.fill(0.0);
	rz += data->f;
	rz.gemv(data->H,x.z,1.0,1.0); // += H*z
	rz.gemv(data->A,x.v,1.0,1.0,true); // += A'*v
	rz.axpy(x.z,sigma);
	rz.axpy(xbar.z,-1.0*sigma);

	// rv = phi(ys,v), ys = y + sigma(x.v - xbar.v)
	for(int i = 0;i<q;i++){
		double ys = x.y(i) + sigma*(x.v(i) - xbar.v(i));
		rv(i) = pfb(ys,x.v(i),alpha);
	}

	z_norm = rz.norm();
	v_norm = rv.norm();
}

void DenseResidual::Copy(const DenseResidual &x){
	rz.copy(x.rz);
	rv.copy(x.rv);
}

void DenseResidual::Fill(double a){
	rz.fill(a);
	rv.fill(a);
}

double DenseResidual::Norm() const{
	return z_norm + v_norm;
}

double DenseResidual::AbsSum(){
	return rz.asum() + rv.asum();
}

double DenseResidual::Merit(){
	double temp = this->Norm();
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

// // DenseLinearSolver *************************************

DenseLinearSolver::DenseLinearSolver(QPsize size){
	n = size.n;
	q = size.q;

	// fb derivatives
	double *a1 = new double[q];
	double *a2 = new double[q];
	double *a3 = new double[q];
	gamma = StaticMatrix(a1,q);
	mus = StaticMatrix(a2,q);
	Gamma = StaticMatrix(a3,q);

	// workspace residuals
	double *b1 = new double[n];
	double *b2 = new double[q];
	r1 = StaticMatrix(b1,n);
	r2 = StaticMatrix(b2,q);

	// workspace hessian
	double *c = new double[n*n];
	K = StaticMatrix(c,n,n);
}

DenseLinearSolver::~DenseLinearSolver(){
	delete[] gamma.data;
	delete[] mus.data;
	delete[] Gamma.data;

	delete[] r1.data;
	delete[] r2.data;

	delete[] K.data;
}

void DenseLinearSolver::LinkData(DenseData *data_){
	this->data = data_;
}

void DenseLinearSolver::SetAlpha(double alpha_){
	alpha = alpha_;
}

bool DenseLinearSolver::Factor(const DenseVariable &x,const DenseVariable &xbar, double sigma){

	// compute K = H + sigma I + A'*Gamma A
	K.copy(data->H);
	K.AddDiag(sigma);
	
	// K <- K + A'*diag(Gamma(x))*A
	Point2D tmp;
	for(int i = 0;i<q;i++){
		double ys = x.y(i) + sigma*(x.v(i) - xbar.v(i));
		tmp = this->PFBgrad(ys,x.v(i),sigma);

		gamma(i) = tmp.x;
		mus(i) = tmp.y + sigma*tmp.x;
		Gamma(i) = gamma(i)/mus(i);
	}
	K.gram(data->A,Gamma);

	// K = LL' in place
	int chol_flag = K.llt();

	if(chol_flag == 0)
		return true;
	else
		return false;
}

bool DenseLinearSolver::Solve(const DenseResidual &r, DenseVariable *x){
	// solve the system
	// K z = rz - A'*diag(1/mus)*rv
	// diag(mus) v = rv + diag(gamma)*A*z

	for(int i = 0;i<mus.rows();i++){
		mus(i) = 1.0/mus(i);
	}

	r1.copy(r.rz);
	r2.copy(r.rv);
	// r2 = rv./mus
	r2.RowScale(mus);
	// r1 = -1.0*A'*r2 + 1.0*r1
	r1.gemv(data->A,r2,-1.0,1.0,true); 

	// solve K z = r1
	r1.CholSolve(K);
	(x->z).copy(r1);

	// r2 = rv + diag(gamma)*A*z
	r2.gemv(data->A,x->z,1.0);
	r2.RowScale(gamma);
	r2.axpy(r.rv,1.0);

	// v = diag(1/mus)*r2
	r2.RowScale(mus);
	(x->v).copy(r2);

	// y = b - Az
	(x->y).copy(data->b);
	(x->y).gemv(data->A,x->z,-1.0,1.0);

	return true;
}


DenseLinearSolver::Point2D DenseLinearSolver::PFBgrad(double a,
 double b, double sigma){
	double y = 0;
	double x = 0;
	double r = sqrt(a*a + b*b);
	double d = 1.0/sqrt(2.0);

	if(r < zero_tol){
		x = alpha*(1.0-d);
		y = alpha*(1.0-d);

	} else if((a > 0) && (b > 0)){
		x = alpha * (1.0- a/r) + (1.0-alpha) * b;
		y = alpha * (1.0- b/r) + (1.0-alpha) * a;

	} else {
		x = alpha * (1.0 - a/r);
		y = alpha * (1.0 - b/r);
	}

	Point2D out = {x, y};
	return out;
}

// DenseFeasibilityCheck *************************************


DenseFeasibilityCheck::DenseFeasibilityCheck(QPsize size){
	n = size.n;
	q = size.q;

	// allocate memory
	double *r1 = new double[n];
	double *r2 = new double[n];
	double *r3 = new double[q];

	z1 = StaticMatrix(r1,n);
	z2 = StaticMatrix(r2,n);
	v1 = StaticMatrix(r3,q);
}

DenseFeasibilityCheck::~DenseFeasibilityCheck(){
	delete[] z1.data;
	delete[] z2.data;
	delete[] v1.data;
}

void DenseFeasibilityCheck::LinkData(DenseData *data){
	int i = 0;
	i++;
}

void DenseFeasibilityCheck::CheckFeasibility(const DenseVariable &x, double tol){

	// check dual
	double w = x.z.infnorm();
	// max(Az)
	v1.gemv(x.data->A,x.z);
	double d1 = v1.max();
	// f'*z
	double d2 = StaticMatrix::dot(x.data->f,x.z);
	// ||Hz||_inf
	z1.gemv(x.data->H,x.z);
	double d3 = z1.infnorm();

	if( (d1 <=0) && (d2 < 0) && (d3 <= tol*w) ){
		dual_ = false;
	}

	// check primal 
	double u = x.v.infnorm();
	// v'*b
	double p1 = StaticMatrix::dot(x.v,x.data->b);
	// ||A'v||_inf
	z2.gemv(x.data->A,x.v,1.0,0.0,true);
	double p2 = x.z.infnorm();
	if( (p1 < 0) && (p2 <= tol*u) ){
		primal_ = false;
	}
}

bool DenseFeasibilityCheck::Dual(){
	return dual_;
}
bool DenseFeasibilityCheck::Primal(){
	return primal_;
}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake

