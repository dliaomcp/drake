#include "drake/solvers/dominic/linalg/static_matrix.h"

#include <cmath>
#include <iostream>


namespace drake {
namespace solvers {
namespace fbstab {

// constructor
StaticMatrix::StaticMatrix(double *mem, int nrows_, int ncols_){
	// TODO: checks to make sure nrows and ncols are > 0
	this->nrows = nrows_;
	this->ncols = ncols_;
	this->nels = nrows_*ncols_;
	this->cap = nrows_*ncols_;
	this->data = mem;
	this->stride = 1;
}

// default constructor
StaticMatrix::StaticMatrix(){
	nrows = 0;
	ncols = 0;
	nels = 0;
	cap = 0;
	data = nullptr;
	stride = 1;
}

// copy constructor, makes a shallow copy (alias)
StaticMatrix::StaticMatrix(const StaticMatrix &A){
	nrows = A.nrows;
	ncols = A.ncols;
	data = A.data;
	nels = A.nels;
	cap = A.cap;
	stride = A.stride;
}

// assignment, makes an alias
StaticMatrix& StaticMatrix::operator=(const StaticMatrix &A){
	nrows = A.nrows;
	ncols = A.ncols;
	data = A.data;
	nels = A.nels;
	cap = A.cap;
	stride = A.stride;

	return *this;
}

// deep copy
void StaticMatrix::copy(const StaticMatrix& A){
	if(cap < A.cap)
		throw std::length_error("Target matrix does not have enough memory avilable to perform the copy operation");

	nrows = A.nrows;
	ncols = A.ncols;
	nels = A.nels;
	stride = A.stride;

	for(int k = 0;k<A.nels;k++){
		data[k] = A.data[k];
	}
}

// map
void StaticMatrix::map(double* mem, int nrows_, int ncols_){
	// TODO: checks to make sure nrows and ncols are > 0
	this->nrows = nrows_;
	this->ncols = ncols_;
	this->nels = nrows_*ncols_;
	this->cap = nrows_*ncols_;
	this->data = mem;
	this->stride = 1;
}

// fill
void StaticMatrix::fill(double a){
	for(int k = 0;k<cap;k++){
		data[k] = a;
	}
}

void StaticMatrix::rand(){
	for(int k = 0;k<cap;k++){
		data[k] = static_cast<double>(std::rand() % 100);
	}
}

void StaticMatrix::eye(){
	StaticMatrix A(*this);
	if(!A.IsSquare()) throw std::invalid_argument("eye can only be called on a square matrix");

	for(int i = 0;i< nrows;i++){
		for(int j = 0;j < ncols;j++){
			if(i==j)
				A(i,j) = 1;
			else
				A(i,j) = 0;
		}
	}
}

// zero out upper triangle
void StaticMatrix::tril(){
	StaticMatrix A(*this);
	if(!A.IsSquare()) throw std::invalid_argument("tril can only be called on a square matrix");

	for(int j =0;j<ncols;j++){
		for(int i = 0;i < j;i++){
			A(i,j) = 0.0;
		}
	}
}

// project all elements into the interval [a,b] 
void StaticMatrix::clip(double a, double b){
	StaticMatrix A(*this);

	for(int k = 0;k<A.size();k++){
		A(k) = max(A(k),a);
		A(k) = min(A(k),b);
	}
}

// size checking
bool StaticMatrix::IsVector() const{
	StaticMatrix x(*this);
	return (x.rows() == 1) || (x.cols() == 1);
}

bool StaticMatrix::IsRow() const{
	StaticMatrix x(*this);
	return (x.rows() == 1);
}

bool StaticMatrix::IsCol() const{
	StaticMatrix x(*this);
	return (x.cols() == 1);
}

bool StaticMatrix::IsSquare() const{
	StaticMatrix x(*this);
	return (x.cols() == x.rows());
}


// matrix indexing
double& StaticMatrix::operator()(int i, int j) const{
	int k = nrows*j + i;

	if((k>= nels)||(i >= nrows)||(j>= ncols) || (i < 0) || (j < 0)) 
		throw std::out_of_range("Index out of bounds");

	return data[k];
}
// vector indexing
double& StaticMatrix::operator()(int i) const{
	StaticMatrix x(*this);
	if(i >= nels || i < 0) throw std::out_of_range("Index out of bounds");
	if(!x.IsVector()) throw std::invalid_argument("Can't use 1D indexing on a Matrix");
	if(i*stride >= cap) throw std::out_of_range("Internal Stride related error");

	return data[stride*i];
}

// get functions
int StaticMatrix::rows() const{
	return nrows;
}
int StaticMatrix::cols() const{
	return ncols;
}
int StaticMatrix::size() const{
	return nels;
}

// set functions
void StaticMatrix::SetStride(int stride_){
	this->stride = stride_;
}
void StaticMatrix::SetCap(int cap_){
	this->cap = cap_;
}

//reshape
void StaticMatrix::reshape(int nrows_,int ncols_){
	// TODO: checks to make sure nrows and ncols are > 0
	if(nrows_*ncols_ > cap)
		throw std::out_of_range("When reshaping the new size cannot exceed capacity");

	this->nrows = nrows_;
	this->ncols = ncols_;
	this->nels = nrows_*ncols_;
}

// return a reshaped alias
StaticMatrix StaticMatrix::getreshape(int nrows_, int ncols_){
	// TODO: checks to make sure nrows and ncols are > 0
	StaticMatrix C(*this);

	if(nrows_*ncols_ != nels)
		throw std::out_of_range("Number of elements cannot change when reshaping");

	// create different sized matrix on top of the memory of the current one
	StaticMatrix A;
	A.map(data,nrows_,ncols_);

	return A;
}

// returns a reference to a column
StaticMatrix StaticMatrix::col(int i){
	// TODO: checks to make sure nrows and ncols are > 0
	StaticMatrix A(*this);
	if(i >= A.cols()) throw std::out_of_range("Requested index exceeds number of columns");
	// offset
	int k = i*A.rows();
	double* ptr = A.data + k;
	StaticMatrix a(ptr,A.rows(),1);

	return a;
}

// returns a reference to a row
StaticMatrix StaticMatrix::row(int i){
	// TODO: checks to make sure nrows and ncols are > 0
	StaticMatrix A(*this);
	if(i >= A.rows()) throw std::out_of_range("Requested index exceeds number of rows");
	// offset
	double *ptr = A.data + i;
	StaticMatrix a(ptr,1,A.cols());
	a.SetStride(A.rows());
	a.SetCap(A.cap);

	return a;
}

StaticMatrix StaticMatrix::subvec(int i, int j){
	StaticMatrix A(*this);
	if(!IsVector()) throw std::runtime_error("subvector cannot be called on a matrix");
	if(i < 0 || j < i || j >= A.size()) throw std::out_of_range("Requested range is out of bounds");

	double *ptr = A.data+i;

	if(IsCol()){
		StaticMatrix a(ptr,j-i+1,1);
		return a;
	} else{
		StaticMatrix a(ptr,1,j-i);
		return a;
	}
}

// y <- a*y
StaticMatrix& StaticMatrix::operator*=(double a){
	StaticMatrix y(*this);

	for(int k=0;k<nels;k++){
		y.data[k] *= a;
	}

	return *this;
}

// y <- y + x
StaticMatrix& StaticMatrix::operator+=(const StaticMatrix &x){
	StaticMatrix y(*this);
	y.axpy(x,1.0);
	return *this;
}

// y <- a*x + y
void StaticMatrix::axpy(const StaticMatrix &x, double a){
	StaticMatrix y(*this);

	if( (y.rows() != x.rows()) || (y.cols() != x.cols()) )
		throw std::length_error("AXPY: Size mismatch");

	for(int i = 0;i<y.rows();i++){
		for(int j = 0;j< y.cols();j++){
			y(i,j) += a*x(i,j);
		}
	}
}

// y <- a*A*x + b*y
void StaticMatrix::gemv(const StaticMatrix &A,const StaticMatrix &x, double a,double b,bool transA){
	if(this == &x){
		#ifdef STATICMATRIX_EXCEPTIONS
		throw std::invalid_argument("A vector can't GEMV itself");
		#endif
	}
	// an alias for easy indexing
	StaticMatrix y(*this);
	int m = y.rows();
	int n = x.rows();

	// input checking
	if(transA){
		if((A.rows() != x.rows()) || (A.cols() != y.rows()))
			throw std::length_error("Inner dimensions must match");
	}
	else{
		if((A.cols() != x.rows()) || (A.rows() != y.rows()))
		 throw std::length_error("Inner dimensions must match");
	}

	if(b != 0.0){
		for(int k =0;k< m;k++){
			y(k) = b*y(k);
		}
	} else{
		y.fill(0.0);
	}

	if(a != 0.0){
		for(int i = 0;i<m;i++){
			for(int j = 0;j<n;j++){
				if(transA)
					y(i) += a*A(j,i)*x(j);
				else
					y(i) += a*A(i,j)*x(j);
			}
		}
	}

}

// C <- a*A*B + b*C
void StaticMatrix::gemm(const StaticMatrix &A, const StaticMatrix &B, double a, double b, bool transA, bool transB){
	int i,j,k;
	int m,n,p;
	StaticMatrix C(*this);

	// input checking
	bool OK = true;
	if( !transA && !transB){ 
		OK = A.cols() == B.rows();
		OK = OK && (A.rows() == C.rows()) && (B.cols() == C.cols());
		m = C.rows();
		n = C.cols();
		p = A.cols();

	} else if(transA && !transB){
		OK = A.rows() == B.rows();
		OK = OK && (A.cols() == C.rows()) && (C.cols() == B.cols());
		m = C.rows();
		n = C.cols();
		p = A.rows();

	} else if(!transA && transB){ 
		OK = A.cols() == B.cols();
		OK = OK && (A.rows() == C.rows()) && (B.rows() == C.cols());
		m = C.rows();
		n = C.cols();
		p = A.cols();

	} else if(transA && transB){ 
		OK = A.rows() == B.cols();
		OK = OK && (A.cols() == C.rows()) && (B.rows() == C.cols());
		m = A.cols();
		n = B.rows();
		p = A.rows();
	} 

	if(!OK){
		#ifdef STATICMATRIX_EXCEPTIONS
		throw std::invalid_argument("Matrix size mismatch in GEMM");
		#endif
	}

	if(b!= 0.0){
		for(i = 0;i<C.rows();i++){
			for(j = 0;j<C.cols();j++){
				C(i,j) = b*C(i,j);
			}
		}
	} else{
		C.fill(0.0);
	}

	if(a!= 0.0){
		for(i = 0;i<C.rows();i++){
			for(j = 0;j<C.cols();j++){
				if(!transA && !transB){ // no transpositions
					for(k = 0;k<p;k++)
						C(i,j) += a*A(i,k)*B(k,j);
							
				} else if(transA && !transB){ // A transposed
					for(k = 0;k<p;k++)
						C(i,j) += a*A(k,i)*B(k,j);

				} else if(!transA && transB){ // B transposed
					for(k = 0;k<p;k++)
						C(i,j) += a*A(i,k)*B(j,k);

				} else if(transA && transB){ // A,B transposed
					for(k = 0;k<p;k++)
						C(i,j) += a*A(k,i)*B(j,k);

				} // trans if
			} // j
		} //i
		
	} // a if
}

// Diagonal Matrix products *************************************
// compute C <- A'*A + C
	void StaticMatrix::gram(const StaticMatrix &A){
		StaticMatrix C(*this);
		bool OK = C.IsSquare();
		OK = OK && (C.rows() == A.cols());
		if(!OK) throw std::invalid_argument("Size Mismatch in gram");

		// call gemm
		C.gemm(A,A,1.0,1.0,true);
	}
	// compute C <-A'*diag(d)*A + C where d is a vector
	void StaticMatrix::gram(const StaticMatrix &A,const StaticMatrix& d){
		StaticMatrix C(*this);
		bool OK = C.IsSquare();
		OK = OK && (C.rows() == A.cols());
		OK = OK && d.IsVector();
		OK = OK && (A.rows() == max(d.cols(),d.rows()));
		if(!OK) throw std::invalid_argument("Size Mismatch in gram");

		int m = C.rows();
		int n = A.rows();

		for(int i = 0;i<m;i++){
			for(int j = 0;j<m;j++){
				for(int k = 0;k<n;k++){
					C(i,j) += A(k,i)*d(k)*A(k,j);
				}
			}
		}


	}
	// compute A <- diag(d)*A where d is a vector
	void StaticMatrix::RowScale(const StaticMatrix &d){
		StaticMatrix A(*this);
		bool OK = d.IsVector();
		int n = max(d.rows(),d.cols());
		OK = OK & (n == A.rows());
		if(!OK) throw std::invalid_argument("Size Mismatch in RScale");

		for(int i = 0; i<A.rows();i++){
			for(int j = 0;j<A.cols();j++){
				A(i,j) *= d(i);
			}
		}

	}
	// compute A <- A*diag(d) where d is a vector
	void StaticMatrix::ColScale(const StaticMatrix &d){
		StaticMatrix A(*this);
		bool OK = d.IsVector();
		int n = max(d.rows(),d.cols());
		OK = OK & (n == A.cols());
		if(!OK) throw std::invalid_argument("Size Mismatch in CScale");

		for(int j = 0;j<A.cols();j++){
			for(int i = 0;i<A.rows();i++){
				A(i,j) *= d(j);
			}
		}
	}


// Norms *************************************
// 2 norm of a vector or Frobenius norm of a matrix
double StaticMatrix::norm() const{
	StaticMatrix A(*this);

	double a = 0;
	for(int i = 0;i<A.rows();i++){
		for(int j = 0;j<A.cols();j++){
			a += A(i,j)*A(i,j);
		}
	}
	return sqrt(a);
}

// sum of absolute values
double StaticMatrix::asum() const{
	StaticMatrix A(*this);

	double a = 0;
	for(int i = 0;i<A.rows();i++){
		for(int j = 0;j<A.cols();j++){
			a += abs(A(i,j));
		}
	}
	return a;
}

double StaticMatrix::infnorm() const{
	StaticMatrix A(*this);

	double a = 0;
	double t = 0;
	for(int i = 0;i<A.rows();i++){
		for(int j = 0;j<A.cols();j++){
			t = abs(A(i,j));
			a = (a >= t ? a : t);
		}
	}
	return a;
}

double StaticMatrix::max() const{
	StaticMatrix A(*this);
	double a = A(0,0);
	for(int i = 0;i<A.rows();i++){
		for(int j = 0;j<A.cols();j++){
			a = max(a,A(i,j));
		}
	}
	return a;
}

double StaticMatrix::min() const{
	StaticMatrix A(*this);
	double a = A(0,0);
	for(int i = 0;i<A.rows();i++){
		for(int j = 0;j<A.cols();j++){
			a = min(a,A(i,j));
		}
	}
	return a;
}

// Factorizations *************************************
int StaticMatrix::llt(){
	StaticMatrix A(*this);
	if(!A.IsSquare())
		throw std::invalid_argument("llt can only be called on a square matrix");

	int n = A.rows();
	bool dyn_reg = false;

	for(int k = 0;k < n;k++){
		double a = A(k,k);
		for(int j = 0;j<k;j++){
			a -= (A(k,j)*A(k,j));
		}
		// modify the factorization if 
		// the matrix is not sufficiently SPD;
		if(a < CHOL_PD_TOL){
			a = CHOL_PD_TOL;
			dyn_reg = true;
		}
		A(k,k) = sqrt(a);
		for(int i = k+1;i < n;i++){
			double b = A(i,k);
			for(int j = 0;j<k;j++){
				b -= A(i,j)*A(k,j);
			}
			A(i,k) = b/A(k,k);
		}
	}

	if(dyn_reg)
		return -1;
	else
		return 0;
}

// Solves LL'x = b in place, i.e., x <- inv(L')*inv(L)*x
void  StaticMatrix::CholSolve(const StaticMatrix &L){
	StaticMatrix x(*this);
	// check L.rows == L.cols == A.rows
	if(!(L.IsSquare()) && (L.cols() == x.rows()))
		throw std::invalid_argument("Matrix size mismatch in LeftCholApply");

	x.LeftCholApply(L); // x <- inv(L)*x
	x.LeftCholApply(L,true); // x <- inv(L')*x
}

// computes A <- A*inv(L) or A <- A*inv(L)'
void StaticMatrix::RightCholApply(const StaticMatrix &L,bool transL){
	
	StaticMatrix A(*this);
	// check that L.rows == L.cols == A.cols
	if(!(L.rows() == L.cols()) && (L.cols() == A.cols()))
		throw std::invalid_argument("Matrix size mismatch in RightCholApply");

	int n = A.cols();
	// loop over rows of A
	for(int i = 0;i<A.rows();i++){
		if(!transL){
			for(int j = n-1;j>=0;j--){
				double b = 0;
				for(int k = n-1;k>j;k--){
					b -= L(k,j)*A(i,k);
				}
				A(i,j) = (A(i,j) + b)/L(j,j);
			}

		} else{
			for(int j = 0;j< n;j++){
				double b = 0;
				for(int k = 0;k <j;k++){
					b -= L(j,k)*A(i,k);
				}
				A(i,j) = (A(i,j) + b)/L(j,j);
			}
		}
	}
}

// computes A <- inv(L)*A or A <- inv(L)'*A
void StaticMatrix::LeftCholApply(const StaticMatrix &L, bool transL){
	StaticMatrix A(*this);
	// check L.rows == L.cols == A.rows
	if(!(L.rows() == L.cols()) && (L.cols() == A.rows()))
		throw std::invalid_argument("Matrix size mismatch in LeftCholApply");

	int n = A.rows();
	// loop over columns
	for(int i = 0;i<A.cols(); i++){
		if(!transL){ // perform a forward solve
			for(int j = 0;j<n;j++){
				double b = 0;
				for(int k = 0;k<= j-1;k++){
					b -= L(j,k)*A(k,i);
				}
				A(j,i) = (A(j,i) + b)/L(j,j);
			}

		} else{ // perform a transposed backsolve
			for(int j = n-1;j>= 0;j--){
				double b = 0;
				for(int k =n-1;k > j;k--){
					b -= L(k,j)*A(k,i);
				}
				A(j,i) = (A(j,i) + b)/L(j,j);
			}
		}
	}
}


// Write to output stream, i.e., print
std::ostream &operator<<(std::ostream& output, const StaticMatrix &A){
	if(!A.IsVector()){
		for(int i=0;i<A.rows();i++){
			for(int j=0;j<A.cols();j++){
				output << A(i,j) << " ";
			}
			output << "\n";
		}
	} else if(A.IsRow()){
		for(int i = 0;i<A.cols();i++){
			output << A(i) << " ";
		}
	} else if(A.IsCol()){
		for(int i = 0;i<A.rows();i++){
			output << A(i) << "\n";
		}
	}
	return output;
}

// dot produce between vectors
double StaticMatrix::dot(const StaticMatrix &y, const StaticMatrix &x){
	bool OK = StaticMatrix::SameSize(x,y);
	OK = OK && y.IsVector();
	OK = OK && x.IsVector();

	if(!OK)
		throw std::invalid_argument("Size mismatch in dot");

	int n = max(x.rows(),x.cols());
	double a = 0;
	for(int i = 0;i<n;i++){
		a+= x(i)*y(i);
	}

	return a;
}

// check if two static matrices are the same size
bool StaticMatrix::SameSize(const StaticMatrix &A, const StaticMatrix &B){
	return (A.rows() == B.rows()) && (A.cols() == B.cols());
}


}  // namespace dominic
}  // namespace solvers
}  // namespace drake






