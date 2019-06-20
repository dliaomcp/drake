#pragma once

#include <Eigen/Dense>

#include "drake/solvers/fbstab/components/dense_variable.h"
#include "drake/solvers/fbstab/components/dense_data.h"

namespace drake {
namespace solvers {
namespace fbstab {

class DenseFeasibility{
 public:
 	DenseFeasibility(int nz, int nv);

 	void LinkData(DenseData *data);
 	void ComputeFeasibility(const DenseVariable &x, double tol);

 	bool IsDualFeasible();
 	bool IsPrimalFeasible();
  private:
  	int nz_ = 0;
  	int nv_ = 0;

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
