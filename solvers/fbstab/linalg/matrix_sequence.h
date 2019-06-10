#pragma once

#include "drake/solvers/fbstab/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {

/**
 * A class that represents sequences of matrices, e.g.,
 * A = A0,A1,A2, ...
 * of the same size.
 * 
 */
class MatrixSequence{
 public:
	/**
	 * Builds a matrix sequence from an array of static matrices.
	 *
	 * @param[in] seq array of StaticMatrix objects
	 * @param[in] nseq length of seq
	 * 
	 */
	MatrixSequence(StaticMatrix *seq, int nseq);

	/**
	 * Build a matrix sequence from continous memory. The memory is organized
	 * by sequence then in column major order, e.g.,
	 *
	 * Q0 = [1,3;2,4], Q1 = [5,7;6,8] would be stored as
	 * seq[] = {1,2,3,4,5,6,7,8}
	 *
	 * @params[in] seq memory
	 * @params[in] nseq length of sequence
	 * @params[in] nrows nummber of rows in the matrices
	 * @params[in] ncols number of columns in the matrices
	 *
	 * length(seq) should be nseq*nrows*ncols
	 * 
	 */
	MatrixSequence(double *mem, int nseq, int nrows, int ncols);

	/**
	 * Build a matrix sequence from an array of arrays. Each matrix should be 
	 * in column major order E.g.
	 * if Q0 = [1,3;2,4], and Q1 = [5,7;6,8] then
	 *
	 * seq[0] should point to {1,2,3,4}
	 * seq[1] should point to {5,6,7,8}
	 *
	 * @params[in] seq memory
	 * @params[in] nseq length of sequence
	 * @params[in] nrows number of rows in the matrices
	 * @params[in] ncols number of columns in the matrices
	 * 
	 */
	MatrixSequence(double **mem, int nseq, int nrows, int ncols);

	/**
	 * Default constructor, initializes everything to 0 or nullptr.
	 */
	MatrixSequence();

	/**
	 * Creates a shallow copy 
	 * @param[in] A reference to the sequence to be copied
	 */
	MatrixSequence(const MatrixSequence &A);

	/**
	 * Calls delete on all the data. 
	 * Assumes that the arrays were allocated with new.
	 * 
	 * Deletes only the top level array if an array of static matrices was supplied
	 * Deletes both nested arrays if an array of arrays was supplied
	 * Deletes the array if a contiguous array was provided
	 */
	void DeleteMemory();

	/**
	 * Assignment operator, shallow copy
	 * @param[in] A reference to the sequence to be copied
	 */
	MatrixSequence& operator=(const MatrixSequence& A);

	/**
	 * Deep copy
	 * @param A Sequence to be copied
	 */
	void copy(const MatrixSequence &A);

	/**
	 * @return number of rows
	 */
	int rows() const;

	/**
	 * @return number of columns
	 */
	int cols() const;

	/**
	 * @return length of the sequence
	 */
	int len() const;

	/**
	 * @return true if nrows == 1 or ncols == 1
	 */
	bool IsVector() const;

	/**
	 * @return true if nrows == 1
	 */
	bool IsRow() const;

	/**
	 * @return true if ncols == 1
	 */
	bool IsCol() const;

	/**
	 * @return true if nrows == ncols
	 */
	bool IsSquare() const;

	/**
	 * Get an alias for the kth matrix in the series.
	 * 
	 * @param  k derised element in the sequence
	 * @return   A StaticMatrix aliasing
	 *
	 * A can be modified to change the sequence!
	 */
	StaticMatrix operator()(int k) const;

	/**
	 * Write to stream.
	 */
	friend std::ostream& operator<<(std::ostream& output, const MatrixSequence &A);

 private:
 	int nrows_; // rows in each matrix
	int ncols_; // columns in each matrix
	int nseq_; // number of matrices in the sequence

	// data
	StaticMatrix *data_ = nullptr;
	double **mem_ = nullptr;
	double *mem1_ = nullptr;
};


}  // namespace fbstab
}  // namespace solvers
}  // namespace drake

