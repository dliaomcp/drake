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

  TimingSimulator(std::string example) {
    // Create the example specified example.
    if (example == "DoubleIntegrator") {
      ocp_.DoubleIntegrator();

    } else if (example == "ServoMotor") {
      ocp_.ServoMotor();

    } else if (example == "SpacecraftRelativeMotion") {
      ocp_.SpacecraftRelativeMotion();

    } else if (example == "CopolymerizationReactor") {
      ocp_.CopolymerizationReactor();

    } else {
      throw runtime_error(example + " is not a valid example.");
    }

    example_name_ = example;
  }

  RunSimulation(bool warmstart_solver = true) {
    // Create the controller object.
    Controller k(ocp_.GetFBstabInput());
    solver_name_ = k.SolverName();

    // Get simulation data.
    SimulationInputs s = ocp_.GetSimulationInputs();
    const int nx = s.B.rows();
    const int nu = s.B.cols();
    const int ny = s.C.rows();

    // Storage for the simulation data,
    // each column is a timestep.
    X.resize(nx, s.T + 1);
    U.resize(nu, s.T + 1);
    Y.resize(ny, s.T + 1);

    // Storage for timing data.
    solve_times_.resize(s.T + 1);
    setup_times_.resize(s.T + 1);
    major_iters_.resize(s.T + 1);
    minor_iters_.resize(s.T + 1);

    // Initial guess for the optimizer.
    Eigen::VectorXd z = Eigen::VectorXd::Zero(ocp.nz());
    Eigen::VectorXd l = Eigen::VectorXd::Zero(ocp.nl());
    Eigen::VectorXd v = Eigen::VectorXd::Zero(ocp.nv());

    // Main simulation loop
    X_.col(0) = x0_;
    for (int i = 0; i < s.T; i++) {
      // Compute control action and record data.
      WrapperOutput w = k.Compute(X.col(i), z, l, v);
      if (warmstart_solver) {
        z = w.z;
        l = w.l;
        v = w.v;
      }
      solve_times_(i) = w.solve_time;
      setup_times_(i) = w.setup_time;
      major_iters_(i) = w.major_iters;
      minor_iters_(i) = w.minor_iters;

      // Propagate system.
      U_.col(i) = w.u;
      Y_.col(i) = s.C * X_.col(i) + s.D * U_.col(i);
      X_.col(i + 1) = s.A * X_.col(i) + s.B * U_.col(i);
    }

    WrapperOutput w = k.Compute(X.col(T), z, l, v);
    U_.col(T) = w.u;
    Y_.col(T) = s.C * X_.col(T) + s.D * U_.col(T);

    solve_times_(T) = w.solve_time;
    setup_times_(T) = w.setup_time;
    major_iters_(T) = w.major_iters;
    minor_iters_(T) = w.minor_iters;

    results_available_ = true;
  }

  bool WriteResultsToFile(std::string file_name) {
    if (!results_available_) {
      throw std::runtime_error(
          "Can't write to file until results are available.");
    }
    const std::ofstream file(filename, std::ios_base::out);
    if (!file.is_open()) {
      return false;
    }
    const int n = solve_times_.size();
    const int nx = X_.rows();
    const int nu = U_.rows();
    const int ny = Y_.rows();

    // Write header.
    file << "Solver: " << solver_name_ << " Example: " << example_name_
         << std::endl;
    file << "t, imajor, iminor, tsolve, tsetup";
    for (int i = 1; i <= nx; i++) {
      file << ", x" + std::to_string(i);
    }

    for (int i = 1; i <= nu; i++) {
      file << ", u" + std::to_string(i);
    }

    for (int i = 1; i <= ny; i++) {
      file << ", y" + std::to_string(i);
    }
    file << std::endl;

    // Loop over timesteps.
    Eigen::IOFormat CSV(Eigen::FullPrecision, Eigen::DontAlignCols, ", ");

    for (int i = 0; i < n; i++) {
      file << i << "," << major_iters_(i) << "," << minor_iters_(i) << ","
           << solve_times_(i) << "," << setup_times_(i);

      file << ", " << X_.col(i).format(CSV);
      file << ", " << U_.col(i).format(CSV);
      file << ", " << Y_.col(i).format(CSV);
      file << std::endl;
    }

    return true;
  }

 private:
  // It can be desirable to average timing data to obtain more reliable
  // timings.
  int num_averaging_steps = 1;

  OcpGenerator ocp_;  // container for the optimal control problem

  std::string example_name_;
  std::string solver_name_;

  Eigen::VectorXd solve_times_;
  Eigen::VectorXd setup_times_;

  // Many solvers have a major-minor iteration structure.
  // Exactly what these mean depends on the solver.
  Eigen::VectorXd major_iters_;
  Eigen::VectorXd minor_iters_;

  // Storage for simulation output.
  Eigen::MatrixXd X_;  // State sequence
  Eigen::MatrixXd U_;  // Control sequence
  Eigen::MatrixXd Y_;  // Output sequence

  bool results_available_ = false;
};

}  // namespace test
}  // namespace fbstab
}  // namespace solvers
}  // namespace drake