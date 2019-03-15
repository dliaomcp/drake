#pragma once

#include "drake/solvers/dominic/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {

class MatrixSequence{
 public:
	// constructor from array of matrix types
	MatrixSequence(StaticMatrix *seq, int nseq);
	MatrixSequence();
	// constructor from contigous memory
	MatrixSequence(double *mem, int nseq, int nrows, int ncols);
	// constructor for array of arrays
	MatrixSequence(double **mem, int nseq, int nrows, int ncols);
	// copy constructor
	MatrixSequence(const MatrixSequence &A);
	// use with care, calls delete
	DeleteMemory();
	// assignment
	MatrixSequence& operator=(const MatrixSequence& A);
	// deep copy
	void copy(const MatrixSequence &A);

	// size
	int rows() const;
	int cols() const;
	int len() const;

	bool IsVector() const;
	bool IsRow() const;
	bool IsCol() const;
	bool IsSquare() const;

	// element access
	double& operator()(int kseq, int i, int j) const;
	// matrix access
	StaticMatrix operator()(int k) const;

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

