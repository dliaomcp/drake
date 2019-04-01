#pragma once

#include <cmath>

#include "drake/solvers/dominic/linalg/static_matrix.h"
#include "drake/solvers/dominic/linalg/matrix_sequence.h"
#include "drake/solvers/dominic/components/mpc_data.h"
#include "drake/solvers/dominic/components/mpc_variable.h"

namespace drake {
namespace solvers {
namespace fbstab {

class MPCFeasibility{
 public:
 	MPCFeasibility(QPsizeMPC size);
 	~MPCFeasibility();
 	void LinkData(MPCData *data);

 	void CheckFeasibility(const MPCVariable &x, double tol);

 	bool Dual();
 	bool Primal();

  private:
  	StaticMatrix z_;
  	StaticMatrix l_;
  	StaticMatrix v_;
  	
  	bool primal_ = true;
  	bool dual_ = true;
  	int nx_, nu_, nc_, N_, nz_, nl_, nv_;
  	MPCData *data_ = nullptr;
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake