#include "drake/solvers/dominic/linalg/matrix_sequence.h"

#include "drake/solvers/dominic/linalg/static_matrix.h"


namespace drake {
namespace solvers {
namespace fbstab {

MatrixSequence::MatrixSequence(StaticMatrix *seq, int nseq){
	data_ = seq;
	nseq_ = nseq;
	nrows_ = data_[0].rows();
	ncols_ = data_[0].cols();
	for(int i=0;i<nseq;i++){
		if(data_[i].rows() != nrows_ || data_[i].cols() != ncols_){
			throw std::runtime_error("Input Sequence must have a uniform size");
		}
	}
}

MatrixSequence::MatrixSequence(){
	nseq_ = 0;
	nrows_ = 0;
	ncols_ = 0;
}

MatrixSequence::MatrixSequence(double *mem, int nseq, int nrows, int ncols){
	nseq_ = nseq;
	ncols_ = ncols;
	nrows_ = nrows;
	mem1_ = mem;
}

MatrixSequence::MatrixSequence(double **mem, int nseq,int nrows, int ncols){
	nseq_ = nseq;
	ncols_ = ncols;
	nrows_ = nrows;
	mem_ = mem;
}

MatrixSequence::MatrixSequence(const MatrixSequence &A){
	nseq_ = A.nseq_;
	nrows_ = A.nrows_;
	ncols_ = A.ncols_;
	data_ = A.data_;
	mem_ = A.mem_;
	mem1_ = A.mem1_;
}

MatrixSequence::DeleteMemory(){
	if(data_ != nullptr){ // free the top level only
		delete[] data_;
	} else if(mem_ != nullptr){
		for(int i = 0;i<nseq_;i++){
			delete[] mem_[i];
		}
		delete[] mem_;
	} else if(mem1_ != nullptr){
		delete[] mem1_;
	}
}

MatrixSequence& MatrixSequence::operator=(const MatrixSequence &A){
	nseq_ = A.nseq_;
	nrows_ = A.nrows_;
	ncols_ = A.ncols_;
	data_ = A.data_;
	mem_ = A.mem_;
	mem1_ = A.mem1_;

	return *this;
}

void MatrixSequence::copy(const MatrixSequence &A){
	MatrixSequence B(*this);
	if(nseq_ != A.nseq_)
		throw std::length_error("Sequences must be the same length to copy");

	for(int k = 0;k<nseq_;k++){
		StaticMatrix C = B(k);
		StaticMatrix D = A(k);
		C.copy(D);
	}
}

int MatrixSequence::rows() const{
	return nrows_;
}

int MatrixSequence::cols() const{
	return ncols_;
}

int MatrixSequence::len() const{
	return nseq_;
}

bool MatrixSequence::IsVector() const{
	return IsRow() || IsCol();
}

bool MatrixSequence::IsCol() const{
	return (nrows_ == 1);
}

bool MatrixSequence::IsRow() const{
	return (ncols_ == 1);
}

bool MatrixSequence::IsSquare() const{
	return (nrows_ == ncols_);
}

StaticMatrix MatrixSequence::operator()(int k) const{
	if(k >= nseq_ || k < 0){
		throw std::length_error("Sequence index out of bounds");
	}

	if(data_ != nullptr){
		return data_[k];

	} else if(mem_ != nullptr){
		StaticMatrix A(mem_[k],nrows_,ncols_);
		return A;

	} else if(mem1_ != nullptr){
		int i = ncols_*nrows_*k;
		StaticMatrix A((mem1_+i),nrows_,ncols_);
		return A;

	} else{
		StaticMatrix A;
		return A;
	}
}

double& MatrixSequence::operator()(int kseq, int i, int j) const{
	StaticMatrix A = this->operator()(kseq);
	return A(i,j);
}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake