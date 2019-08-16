#pragma once

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "drake/common/drake_copyable.h"
#include "drake/solvers/fbstab/test/ocp_generator.h"
#include "drake/solvers/fbstab/test/solver_wrappers.h"

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
    if (example == "DoubleIntegrator") {
      ocp_.DoubleIntegrator();

    } else if (example == "ServoMotor") {
      ocp_.ServoMotor();

    } else if (example == "SpacecraftRelativeMotion") {
      ocp_.SpacecraftRelativeMotion();

    } else if (example == "CopolymerizationReactor") {
      ocp_.CopolymerizationReactor();

    } else {
      throw std::runtime_error(example + " is not a valid example.");
    }

    example_name_ = example;
  }

  void SetAveragingSteps(int nave) { num_averaging_steps_ = nave; }

  void RunSimulation(bool warmstart_solver = true) {
    // Create the controller object.
    Controller k(ocp_.GetFBstabInput());
    solver_name_ = k.SolverName();

    std::cout << "Running: " << example_name_ << ", Solver: " << solver_name_
              << " ...";

    // Get simulation data.
    OcpGenerator::SimulationInputs s = ocp_.GetSimulationInputs();
    const int nx = s.B.rows();
    const int nu = s.B.cols();
    const int ny = s.C.rows();

    // Storage for the simulation data,
    // each column is a timestep.
    X_.resize(nx, s.T + 1);
    U_.resize(nu, s.T + 1);
    Y_.resize(ny, s.T + 1);

    // Storage for timing data.
    solve_times_.resize(s.T + 1);
    setup_times_.resize(s.T + 1);
    major_iters_.resize(s.T + 1);
    minor_iters_.resize(s.T + 1);
    residuals_.resize(s.T + 1);
    initial_residuals_.resize(s.T + 1);
    success_.resize(s.T + 1);

    // Initial guess for the optimizer.
    Eigen::VectorXd z = Eigen::VectorXd::Zero(ocp_.nz());
    Eigen::VectorXd l = Eigen::VectorXd::Zero(ocp_.nl());
    Eigen::VectorXd v = Eigen::VectorXd::Zero(ocp_.nv());

    Eigen::VectorXd temp1(num_averaging_steps_);
    Eigen::VectorXd temp2(num_averaging_steps_);

    // Main simulation loop
    X_.col(0) = s.x0;

    for (int i = 0; i < s.T; i++) {
      // Compute control action and record data.
      WrapperOutput w;

      for (int j = 0; j < num_averaging_steps_; j++) {
        w = k.Compute(X_.col(i), z, l, v);
        temp1(j) = w.solve_time;
        temp2(j) = w.setup_time;
      }

      solve_times_(i) = temp1.mean();
      setup_times_(i) = temp2.mean();
      major_iters_(i) = w.major_iters;
      minor_iters_(i) = w.minor_iters;
      residuals_(i) = w.residual;
      success_(i) = w.success ? 1 : 0;
      initial_residuals_(i) = w.initial_residual;

      if (warmstart_solver) {
        z = w.z;
        l = w.l;
        v = w.v;
      }

      // Propagate system.
      U_.col(i) = w.u;
      Y_.col(i) = s.C * X_.col(i) + s.D * U_.col(i);
      X_.col(i + 1) = s.A * X_.col(i) + s.B * U_.col(i);
    }

    WrapperOutput w = k.Compute(X_.col(s.T), z, l, v);
    U_.col(s.T) = w.u;
    Y_.col(s.T) = s.C * X_.col(s.T) + s.D * U_.col(s.T);

    solve_times_(s.T) = w.solve_time;
    setup_times_(s.T) = w.setup_time;
    major_iters_(s.T) = w.major_iters;
    minor_iters_(s.T) = w.minor_iters;
    residuals_(s.T) = w.residual;
    initial_residuals_(s.T) = w.initial_residual;
    success_(s.T) = w.success ? 1 : 0;

    results_available_ = true;

    std::cout << " done." << std::endl;
  }

  Eigen::MatrixXd GetState() { return X_; }
  Eigen::MatrixXd GetControl() { return U_; }
  Eigen::MatrixXd GetOutput() { return Y_; }

  bool WriteResultsToFile(std::string file_name) {
    if (!results_available_) {
      throw std::runtime_error(
          "Can't write to file until results are available.");
    }
    std::ofstream file(file_name, std::ios_base::out);
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
    file << "t, imajor, iminor, tsolve, tsetup, RES, RES0, SUCCESS";
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
           << solve_times_(i) << "," << setup_times_(i) << "," << residuals_(i)
           << "," << initial_residuals_(i) << "," << success_(i);

      file << ", " << X_.col(i).transpose().format(CSV);
      file << ", " << U_.col(i).transpose().format(CSV);
      file << ", " << Y_.col(i).transpose().format(CSV);
      file << std::endl;
    }

    return true;
  }

 private:
  // It can be desirable to average timing data to obtain more reliable
  // timings.
  int num_averaging_steps_ = 1;

  OcpGenerator ocp_;  // container for the optimal control problem

  std::string example_name_;
  std::string solver_name_;

  Eigen::VectorXd solve_times_;
  Eigen::VectorXd setup_times_;

  Eigen::VectorXd residuals_;
  Eigen::VectorXd initial_residuals_;
  Eigen::VectorXi success_;

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