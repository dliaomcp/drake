#include "drake/solvers/fbstab/components/dense_data.h"

#include <cmath>

#include "drake/solvers/fbstab/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {

DenseData::DenseData(double *H,double *f, 
	double *A,double *b, DenseQPsize size){

	n_ = size.n;
	q_ = size.q;

	// create static matrices over the inputs
	H_.map(H,n_,n_);
	f_.map(f,n_);
	A_.map(A,q_,n_);
	b_.map(b,q_);
}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake

