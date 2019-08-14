#pragma once

#include <cmath>
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

/**
 * Create a vector of (approximately since they're integers) logarithmically
 * spaced vectors on the interval [10^a,10^b]. E.g., logspace(0,3,4) =
 * [1,10,100,1000].
 *
 * @param[in]  a
 * @param[in]  b
 * @param[in]  n number of points
 * @return   logspace vector
 */
Eigen::VectorXi logspace(double a, double b, int n) {
  // Create a logspaced vector of doubles.
  Eigen::VectorXd l = Eigen::VectorXd::LinSpaced(n, a, b);
  Eigen::VectorXd ten = 10 * Eigen::VectorXd::Ones(n);
  l = ten.array().pow(l.array());

  // Round to the nearest integer.
  Eigen::VectorXi out = Eigen::VectorXi::Zero(l.size());
  for (int i = 0; i < l.size(); i++) {
    out(i) = static_cast<int>(ceil(l(i)));
  }

  return out;
}

/**
 * This class implements prediction horizon scaling benchmarking/timing for
 * optimal control problems.
 *
 * @tparam SolverWrapper This class wraps the solver with a unified interface to
 * enable easy testing of multiple solvers.
 */
template <class SolverWrapper>
class ScalingBenchmark {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ScalingBenchmark)

  /**
   * Sets up the scaling benchmark, i.e., creates a series of optimal control
   * problems of increasing dimension. This class uses OcpGenerator to create
   * the OCPs, see ocp_generator.h for details about which problems are
   * available.
   *
   * @param[in] N vector of horizon lengths, e.g., [10,100,100]
   * @param[in] example, a string corresponding to any of the problem creation
   * methods in OcpGenerator, e.g., "ServoMotor".
   *
   * Throws a runtime_error if N.size() == 0, any min(N) <= 0 or if example
   * doesn't correspond to a valid method in OcpGenerator.
   */
  ScalingBenchmark(const Eigen::VectorXi& N, std::string example) {
    if (N.minCoeff() <= 0) {
      throw std::runtime_error(
          "In ScalingBenchmark::ScalingBenchmark: All entries in N must be > "
          "0.");
    }
    if (N.size() == 0) {
      throw std::runtime_error(
          "In ScalingBenchmark::ScalingBenchmark: length(N) == 0.");
    }

    // Create the examples.
    int n = N.size();
    problem_instances_.reserve(n);

    for (int i = 0; i < n; i++) {
      problem_instances_.push_back(OcpGenerator());

      if (example == "DoubleIntegrator") {
        problem_instances_[i].DoubleIntegrator(N(i));

      } else if (example == "ServoMotor") {
        problem_instances_[i].ServoMotor(N(i));

      } else if (example == "SpacecraftRelativeMotion") {
        problem_instances_[i].SpacecraftRelativeMotion(N(i));

      } else if (example == "CopolymerizationReactor") {
        problem_instances_[i].CopolymerizationReactor(N(i));

      } else {
        throw std::runtime_error(example + " is not a valid example.");
      }
    }

    example_name_ = example;
    t_average_.resize(n);
    t_max_.resize(n);
    t_std_.resize(n);
    iters_.resize(n);
    residual_.resize(n);
    success_.resize(n);

    horizon_length_ = N;
    num_averaging_steps_ = Eigen::VectorXi::Ones(n);
  }

  // runs the timing and stores the result internally
  void RunTiming() {
    int n = horizon_length_.size();

    for (int i = 0; i < n; i++) {
      // Create the controller.
      OcpGenerator& ocp = problem_instances_.at(i);
      FBstabMpc::QPData data = ocp.GetFBstabInput();
      SolverWrapper solver(data);
      if (i == 0) {
        solver_name_ = solver.SolverName();
      }

      // Create initial guess, for scaling tests everything is initialized at
      // the origin.
      Eigen::VectorXd z0 = Eigen::VectorXd::Zero(ocp.nz());
      Eigen::VectorXd l0 = Eigen::VectorXd::Zero(ocp.nl());
      Eigen::VectorXd v0 = Eigen::VectorXd::Zero(ocp.nv());

      Eigen::VectorXd texe(num_averaging_steps_(i));

      std::cout << "Running: " << solver.SolverName()
                << " N = " << horizon_length_(i);
      std::cout << " ...";
      for (int j = 0; j < num_averaging_steps_(i); j++) {
        // Call the solver, for these tests we use the default value of x0 given
        // by the OCP generator.
        WrapperOutput out = solver.Compute(*data.x0, z0, l0, v0);
        texe(j) = out.solve_time;
        if (j == 0) {
          iters_(i) = out.major_iters;
          residual_(i) = out.residual;
          success_(i) = out.success ? 1 : 0;
        }
        if (out.success == false) {
          std::cout << "Warning: Solver " + solver.SolverName() + " failed.";
          // No use continuing, just repeat the existing number.
          texe = Eigen::VectorXd::Ones(texe.size()) * texe(0);
          break;
        }
      }

      std::cout << " done." << std::endl;
      // Compute execution time statistics.
      t_average_(i) = texe.mean();
      t_max_(i) = texe.maxCoeff();

      t_std_(i) = 0.0;
      if (num_averaging_steps_(i) > 1) {
        for (int j = 0; j < num_averaging_steps_(i); j++) {
          t_std_(i) += pow(texe(j) - t_average_(i), 2);
        }
        t_std_(i) /= (num_averaging_steps_(i) - 1);
        t_std_(i) = sqrt(t_std_(i));
      }
    }

    data_available_ = true;
  }

  void UpdateAveragingVector(Eigen::VectorXi nave) {
    if (nave.size() != horizon_length_.size()) {
      throw std::runtime_error("nave.size() != N.size()");
    }
    num_averaging_steps_ = nave;
  }

  /**
   * Writes stored results to the specified text file in CSV format.
   *
   * @param[in] filename
   * @return true if the operation was successful, false otherwise
   */
  bool WriteResultsToFile(std::string filename) {
    if (!data_available_) {
      throw std::runtime_error("Can't write non-existent data to file.");
    }

    std::ofstream file(filename, std::ios_base::out);
    if (!file.is_open()) {
      return false;
    }

    // Write header.
    file << "Solver: " << solver_name_ << " Example: " << example_name_
         << std::endl;
    file << "Horizon Length, TAVE, TMAX, TSTD, RES, ITER, SUCCESS" << std::endl;

    // Concatenate the data then use Eigen formatting tools.
    // T = [tave,tmax,tstd].
    Eigen::MatrixXd T(horizon_length_.size(), 7);
    T.col(0) = horizon_length_.cast<double>();
    T.col(1) = t_average_;
    T.col(2) = t_max_;
    T.col(3) = t_std_;
    T.col(4) = residual_;
    T.col(5) = iters_.cast<double>();
    T.col(6) = success_.cast<double>();

    Eigen::IOFormat CSV(Eigen::FullPrecision, 0, ", ");
    file << T.format(CSV);
    file.close();

    return true;
  }

  Eigen::VectorXd GetAverageResults() {
    // TODO(dliaomcp@umich.edu) Check that data is ready.
    return t_average_;
  }

  Eigen::VectorXd GetMaxResults() { return t_max_; }

  Eigen::VectorXd GetStdResults() { return t_std_; }

 private:
  std::vector<OcpGenerator> problem_instances_;
  Eigen::VectorXi num_averaging_steps_;
  Eigen::VectorXi horizon_length_;
  std::string example_name_;
  std::string solver_name_;

  // Average, max and standard deviations.
  Eigen::VectorXd t_average_;
  Eigen::VectorXd t_max_;
  Eigen::VectorXd t_std_;
  Eigen::VectorXi iters_;
  Eigen::VectorXd residual_;
  Eigen::VectorXi success_;

  bool data_available_ = false;
};

}  // namespace test
}  // namespace fbstab
}  // namespace solvers
}  // namespace drake