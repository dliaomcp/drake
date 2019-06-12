#pragma once

#include "drake/solvers/fbstab/linalg/static_matrix.h"


// contains the gereral dense qp specific classes and structures

namespace drake {
namespace solvers {
namespace fbstab {


// stores the size of the qp
struct DenseQPsize {
	int n; // primal dimension
	int q; // number of inequalities
};

// stores the problem data
class DenseData{
 public:
	DenseData(double *H,double *f, double *A,double *b, DenseQPsize size);

 private:
 	StaticMatrix H_,f_,A_,b_;
	int n_,q_;
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
