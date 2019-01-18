#pragma once
#include "static_matrix.h"

// template this on matrix type eventually?
class MatrixSequence{

public:

	// properties *************************************
	int nrows; // rows in each matrix
	int ncols; // columns in each matrix
	int nseq; // number of matrices in the sequence
	double **data; // array of pointers to the raw
	StaticMatrix


	// methods *************************************

	// constructor from array of matrix types
	MatrixSequence(StaticMatrix *seq);
	// constructor from raw memory
	MatrixSequence(double *mem, int nrows, int ncols,int nseq);



	int rows() const;
	int cols() const;
	int len() const;

	// element access

	// matrix access

	// multiplcation?

};