#pragma once

#include <Eigen/Dense>

#include "drake/solvers/fbstab/linalg/static_matrix.h"
#include "drake/solvers/fbstab/components/dense_variable.h"
#include "drake/solvers/fbstab/components/dense_data.h"

namespace drake {
namespace solvers {
namespace fbstab {

class DenseFeasibility{
 public:
 	DenseFeasibility(int n, int q);

 	void LinkData(DenseData *data);
 	void ComputeFeasibility(const DenseVariable &x, double tol);

 	bool IsDualFeasible();
 	bool IsPrimalFeasible();
  private:
  	int n_,q_;

  	// workspace
  	Eigen::VectorXd z1_;
  	Eigen::VectorXd v1_;

  	bool primal_feasible_ = true;
  	bool dual_feasible_ = true;
  	DenseData *data_ = nullptr;
};


}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
