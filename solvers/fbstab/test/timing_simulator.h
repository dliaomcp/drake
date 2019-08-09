#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>

#include "drake/common/drake_copyable.h"
#include "drake/solvers/fbstab/test/ocp_generator.h"

namespace drake {
namespace solvers {
namespace fbstab {
namespace test {

// TODO(dliaomcp@umich.edu) Figure out a good way to ID sims.

/**
 * This class is used to perform closed-loop simulations
 * of MPC controllers using various solvers for benchmarking purposes.
 *
 * Its templated on a controller object which wraps the solver
 */
template <class Controller>
class TimingSimulator {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(TimingSimulator)

  TimingSimulator(const OCPGenerator::SimulationInputs s) {
    A_ = s.A;
    B_ = s.B;
    C_ = s.C;
    D_ = s.D;
    x0_ = s.x0;
  }

  SetInitialCondition(const Eigen::VectorXd& x0) { x0_ = x0; }

  SetNumberOfSteps(int T) { T_ = T; }

  // Simulate(){X.reserve()}

  WriteResultToFile(string file_name);

 private:
  // It can be desirable to average timing data to obtain more reliable
  // timings.
  int num_averaging_steps = 1;

  // Problem data
  Eigen::MatrixXd A_;
  Eigen::MatrixXd B_;
  Eigen::MatrixXd C_;
  Eigen::MatrixXd D_;
  Eigen::MatrixXd x0_;
  int T_ = 0;  // Number of timestep in the simulation.

  Eigen::VectorXd solve_times_;
  Eigen::VectorXd setup_times_;

  // Many solvers have a major-minor iteration structure.
  // Exactly what these mean depends on the solver.
  Eigen::VectorXd major_iters_;
  Eigen::VectorXd minor_iters_;

  // Storage for simulation output.
  std::vector<Eigen::VectorXd> X;  // State sequence
  std::vector<Eigen::VectorXd> U;  // Control sequence
  std::vector<Eigen::VectorXd> Y;  // Output sequence

  bool results_available_ = false;
};

}  // namespace test
}  // namespace fbstab
}  // namespace solvers
}  // namespace drake