#pragma once

#include <cmath>

#include "drake/solvers/dominic/linalg/static_matrix.h"
#include "drake/solvers/dominic/linalg/matrix_sequence.h"
#include "drake/solvers/dominic/components/mpc_data.h"
#include "drake/solvers/dominic/components/mpc_variable.h"

namespace drake {
namespace solvers {
namespace fbstab {

/**
 * A class that computes and store infeasibility certificates for MPC QPs.
 */
class MPCFeasibility{
 public:
 	
 	/**
 	 * Allocates workspace memory.
 	 */
 	MPCFeasibility(QPsizeMPC size);

 	/**
 	 * Frees allocated memory.
 	 */
 	~MPCFeasibility();

 	/**
 	 * Links to problem data needed to perform calculations
 	 * Calculations cannot be performed until a data object is provided
 	 * @param[in] data Pointer to the problem data
 	 */
 	void LinkData(MPCData *data);

 	/**
 	 * Checks to see if x is an infeasibility certificate for the QP and stores the 
 	 * result internally.
 	 * @param[in] x   infeasibility certificate candidate
 	 * @param[in] tol numerical tolerance used when checking the conditions
 	 */
 	void CheckFeasibility(const MPCVariable &x, double tol);

 	/**
 	 * Retrieve the result of the last infeasibility check
 	 * @return false if a dual infeasibility certificate was found, true otherwise
 	 */
 	bool Dual();

 	/**
 	 * Retrieve the result of the last infeasibility check
 	 * @return false if a primal infeasibility certificate was found, true otherwise
 	 */
 	bool Primal();

  private:
  	// workspaces
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