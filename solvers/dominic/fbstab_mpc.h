#pragma once

#include "drake/solvers/dominic/linalg/static_matrix.h"
#include "drake/solvers/dominic/linalg/matrix_sequence.h"

namespace drake {
namespace solvers {
namespace fbstab {

struct QPDataMPC {
	double **Q = nullptr;
	double **R = nullptr;
	double **S = nullptr;
	double **q = nullptr;
	double **r = nullptr;

	double **A = nullptr;
	double **B = nullptr;
	double **c = nullptr;
	double **x0 = nullptr;

	double **E = nullptr;
	double **L = nullptr;
	double **d = nullptr;
};

class FBstabMPC {

};


}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
