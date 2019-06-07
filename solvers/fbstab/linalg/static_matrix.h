#pragma once

#include <iostream>


// if STATICMATRIX_EXCEPTIONS is defined then
// matrix operations in the static matrix class will throw
// exceptions
#define STATICMATRIX_EXCEPTIONS
#define CHOL_PD_TOL 1e-13


namespace drake {
namespace solvers {
namespace fbstab {

/**
 * A class which "wraps" simple arrays to allow for matrix operations.
 * No memory is ever allocated or deleted in this class. 
 * All matrices are stored in column major order.
 */
class StaticMatrix{
public:
	/**
	 * Pointer to the underlying memory
	 * TODO: make me private
	 */
	double *data;
	
	/**
	 * Builds a StaticMatrix on existing memory.
	 *
	 * @param[in] mem pointer to an array.
	 * @param[in] nrows_ number of rows.
	 * @param[in] ncols_ number of columns.
	 *
	 * StaticMatrix uses column major format.
	 * Ensure that sizeof(mem) = nrows_*ncols_*sizeof(double)
	 * Otherwise segfaults will result.
	 */
	StaticMatrix(double* mem, int nrows_, int ncols_ = 1);
	/**
	 * Creates a matrix, everything is 0 or nullptr
	 */
	StaticMatrix();

	/**
	 * Copy constructor. Creates a shallow copy of A.
	 */
	StaticMatrix(const StaticMatrix &A);

	/**
	 * Fills this with a.
	 * @param[in] a value to fill the matrix
	 */
	void fill(double a);

	/**
	 * Overwrites this with the a*I.
	 * @param[in] a scaling factor
	 *
	 * this must be square 
	 */
	void eye(double a = 1.0);

	/**
	 * Overwrites this with a matrix of random numbers.
	 */
	void rand();

	/**
	 * Zeros out the upper triangle.
	 */
	void tril();

	/**
	 * Projects all elements of this into [a,b]
	 * @param[in] a lower bound				
	 * @param[in] b upper bound
	 */
	void clip(double a,double b);

	// Size checks *************************************
	/**
	 * Self-explanatory
	 */
	bool IsVector() const;
	bool IsRow() const;
	bool IsCol() const;
	bool IsSquare() const;

	//  Geters *************************************
	/**
	 * @return number of rows
	 */
	int rows() const;

	/**
	 * @return number of columns
	 */
	int cols() const;

	/**
	 * @return Total number of elements (rows*cols)
	 */
	int size() const;

	/**
	 * @return Total number of elements (rows*cols)
	 */
	int len() const;

	// Operator overloads *************************************
	/**
	 * 2D (matrix) indexing
	 * @param[in]  i row index
	 * @param[in]  j column index
	 * @return   reference to the jth element of the ith row
	 *
	 * TODO: this should not be const 
	 * Need to remove without breaking ability to read const
	 * StaticMatrices
	 */
	double& operator()(int i,int j) const;

	/**
	 * 1D (vector) indexing
	 * @param[in]  i index
	 * @return   reference to the ith element of a vector
	 *
	 * TODO: this should not be const 
	 * Need to remove without breaking ability to read const
	 * StaticMatrices
	 */
	double& operator()(int i) const;

	/**
	 * Assignment (shallow copy)
	 */
	StaticMatrix& operator=(const StaticMatrix& A);

	/**
	 * Scales the matrix by a, i.e., y <- a*y 
	 */
	StaticMatrix& operator*=(double a);

	/**
	 * Adds a matrix, i.e., y <- x+y
	 */
	StaticMatrix& operator+=(const StaticMatrix& x);

	// Slicing and mapping *************************************

	/**
	 * Maps a static matrix onto existing memory.
	 * Essentially a constructor call.
	 * 
	 * @param[in] mem pointer to an array.
	 * @param[in] nrows_ number of rows.
	 * @param[in] ncols_ number of columns.
	 *
	 * StaticMatrix uses column major format.
	 * Ensure that sizeof(mem) = nrows_*ncols_*sizeof(double)
	 * Otherwise segfaults will result.
	 */
	void map(double* mem, int nrows_, int ncols_ = 1);

	/**
	 * Reshapes an StaticMatrix in place without modifying its elements.
	 * @param[in] nrows_ new number of rows
	 * @param[in] ncols_ new number of columns
	 *
	 * When calling reshape the produce nrows*ncols cannot change.
	 * StaticMatrix uses column major ordering so e.g.,
	 *
	 * [1,3] .reshape(4,1) = [1]
	 * [2,4]    			 [2]
	 * 						 [3]
	 * 						 [4]
	 * 						 
	 */
	void reshape(int nrows_, int ncols_);

	/**
	 * Get an reshaped matrix that aliases this
	 * @param[in]  nrows_ new number of rows
	 * @param[in]  ncols_ new number of columns
	 * @return        Static matrix of size nrows_*ncols_ 
	 *
	 * The return value references the same underlying memory!
	 */
	StaticMatrix getreshape(int nrows_, int ncols_) const;

	/**
	 * Get an alis for the ith column
	 * @param[in]  i desired column
	 * @return     a StaticMatrix that aliases the ith column of this
	 */
	StaticMatrix col(int i);

	/**
	 * Get an alias for the ith row
	 * @param[in]  i desired row
	 * @return     a StaticMatrix that aliases the ith row of this
	 */
	StaticMatrix row(int i);

	/**
	 * Get an alias for a portion of a vector e.g.,
	 * if x = {1,2,3,4}
	 * x.subvec(1,2) returns an alias to {2,3}
	 * 
	 * @param[in]  i start index
	 * @param[in]  j end index
	 * @return   a StaticMatrix that aliases x(i:j)
	 */
	StaticMatrix subvec(int i, int j);

	/**
	 * Make a deep copy of a matrix
	 * @param[in] A matrix to be copied
	 */
	void copy(const StaticMatrix& A);

	// BLAS operations *************************************

	/**
	 * Computes y <- a*x + y where y = *this is a vector
	 * 
	 * @param[in] x vector
	 * @param[in] a scaling factor
	 *
	 * axpy can't be performed "in place", an error is thrown if &x == this.
	 */
	void axpy(const StaticMatrix &x, double a);

	/**
	 * Computes y <- a*P*x+b*y where y = *this is a vector and
	 * P can be A or A' 
	 * 
	 * @param[in] A      Matrix input
	 * @param[in] x      Vector input
	 * @param[in] a      scaling factor
	 * @param[in] b      self-scaling factor
	 * @param[in] transA If true P = A', otherwise P = A
	 *
	 * gemv can't be performed "in place", an error is thrown if &x == this.
	 * 
	 */
	void gemv(const StaticMatrix &A, const StaticMatrix &x, double a = 1.0, double b = 0.0, bool transA = false);

	/**
	 * Computes C <- a*P*G + b*C where C = *this is a matrix and
	 * P = A or A' and G = B or B'
	 * 
	 * @param[in] A      Matrix input
	 * @param[in] B      Matrix input
	 * @param[in] a      Scaling factor
	 * @param[in] b      self-scaling
	 * @param[in] transA If true P = A', otherwise P = A
	 * @param[in] transB If true G = B', otherwise G = B
	 *
	 * gemm can't be performed "in place", an error is thrown 
	 * if &A == this or &B == this.
	 */
	void gemm(const StaticMatrix &A, const StaticMatrix &B, double a, double b, bool transA = false, bool transB = false);

	// Diagonal Matrix products *************************************

	/**
	 * Computes C <- a* A'*A + C where C = *this is a matrix
	 * 
	 * @param[in] A      Matrix input
	 * @param[in] a      scaling factor
	 * @param[in] transA if true C <- a A*A' + C is computed
	 * 
	 */
	void gram(const StaticMatrix &A, double a = 1.0, bool transA = false);

	/**
	 * Computes C <- A'*diag(d)*A + C where C = *this is a matrix
	 * 
	 * @param[in] A      Matrix input
	 * @param[in] d      Vector of scaling factors
	 * 
	 */
	void gram(const StaticMatrix &A,const StaticMatrix& d);

	/**
	 * Scales the rows of A = *this by the elements of d
	 * i.e., A <- diag(d)*A
	 * @param[in] d vector of scaling factors
	 */
	void RowScale(const StaticMatrix &d);

	/**
	 * Scales the columns of A = *this by the elements of d
	 * i.e., A <- A*diag(d)
	 * @param[in] d vector of scaling factors
	 */
	void ColScale(const StaticMatrix &d);

	/**
	 * Adds a diagonal matrix to C = *this
	 * i.e., C <- C + diag(d)
	 * @param[in] d Vector of to add to the diagonal
	 */
	void AddDiag(const StaticMatrix &d);

	/**
	 * Adds a multiple of the identity to C = *this
	 * i.e., C <- C + a*I
	 * @param[in] a scaling factor for the identity 
	 */
	void AddDiag(double a);


	// norms *************************************
	/**
	 * Euclidean Norm
	 * @return 2 norm of a vector and Frobenius norm of a matrix
	 */
	double norm() const; 

	/**
	 * L1 Norm
	 * @return the sum of absolute vales
	 */
	double asum() const;

	/**
	 * Infinity Norm
	 * @return largest absolute value
	 */
	double infnorm() const; 

	/**
	 * Max
	 * @return value of the largest element
	 */
	double max() const;

	/**
	 * Min
	 * @return value of the smallest element
	 */
	double min() const;

	// Factorizations *************************************
	
	/**
	 * Computes the Cholesky factorization of A = *this in place
	 * i.e., A <- L where L is lower triangular and A = L*L'.
	 * 
	 * Only the lower triangle of A is accessed.
	 * If A is not positive definite then dynamic regularization is applied
	 * to force it to be.
	 * 
	 * @return  if successful, -1 if dynamic regularization was applied
	 *
	 * The upper triangle is untouched
	 * and is not zeroed out. 
	 *
	 * TODO, add an option to stop and throw an error or a flag if
	 * the factorization process fails
	 */
	int llt();

	/**
	 * Solve a linear system of equations using a Cholesky factorization.
	 * Applies x <- inv(A)*x using L where A = L*L' and x = *this.
	 * 
	 * @param[in] L StaticMatrix containing the Cholesky factor of A, 
	 * only the lower triangle of L is accessed.
	 */
	void CholSolve(const StaticMatrix &L);

	/**
	 * Apply an inverse Cholesky factorization to A = *this from the right
	 * i.e., compute A <- A*inv(L) or A <- A*inv(L)'
	 * 
	 * @param[in] L      Cholesky factor to be applied, once the lower 
	 * 					 triangle is accessed.
	 * 					 
	 * @param[in] transL If true then L is transposed
	 * 
	 */
	void RightCholApply(const StaticMatrix &L, bool transL = false);

	/**
	 * Apply an inverse Cholesky factorization to A = *this from the left
	 * i.e., compute A <- inv(L)*A or A <- inv(L)'*A
	 * 
	 * @param[in] L      Cholesky factor to be applied, once the lower 
	 *               	 triangle is accessed.
	 * 
	 * @param[in] transL If true then L is transposed
	 * 
	 */
	void LeftCholApply(const StaticMatrix &L, bool transL = false);


	// static methods *************************************
	
	/**
	 * Templated max operation
	 */
	template <class T>
	static T max(T a, T b){
		return (a>b) ? a : b;
	}

	/**
	 * Templated min operation
	 */
	template <class T>
	static T min(T a, T b){
		return (a>b) ? b : a;
	}

	/**
	 * Computes the dot product between two vectors of the same size
	 * i.e., y'*x
	 * 
	 * @param[in]  y Vector 1
	 * @param[in]  x Vector 1
	 * @return   y'*x
	 */
	static double dot(const StaticMatrix& y, const StaticMatrix& x);
	
	/**
	 * Returns if two StaticMatrix objects are the same size
	 * @param[in]  A 
	 * @param[in]  B 
	 * @return   true if A and B have the same numbers of rows and columns
	 *           false otherwise
	 */
	static bool SameSize(const StaticMatrix &A, const StaticMatrix &B);

	/**
	 * Write to a stream.
	 */
	friend std::ostream &operator<<(std::ostream& output, const StaticMatrix &A);

 private:
 	// sizes
	int nrows;
	int ncols;
	int nels;
	int cap;
	int stride;

	// Seters *************************************
	void SetStride(int stride_);
	void SetCap(int cap_);
};


}  // namespace dominic
}  // namespace solvers
}  // namespace drake





