#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "drake/common/drake_copyable.h"
#include "drake/solvers/fbstab/test/ocp_generator.h"

namespace drake {
namespace solvers {
namespace fbstab {
namespace test {

template <class SolverWrapper>
class ScalingBenchmark {
 public:
  // example can be any of the ones in ocp generator
  ScalingBenchmark(Eigen::VectorXi N, string example) {
    int n = N.size();
    horizon_length_ = N;
    num_averaging_steps_ = Eigen::VectorXi::Ones(n);
    problem_instances_.reserve(n);

    t_average_.resize(n);
    t_max_.resize(n);
    t_std_.resize(n);

    // Create the examples.
    for (int i = 0; i < n; i++) {
      problem_instances_.push_back(OCPGenerator());

      if (example == "DoubleIntegrator") {
        problem_instances_[i].DoubleIntegrator(N(i));

      } else if (example == "ServoMotor") {
        problem_instances_[i].ServoMotor(N(i));

      } else if (example == "SpacecraftRelativeMotion") {
        problem_instances_[i].SpacecraftRelativeMotion(N(i));

      } else if (example == "CopolymerizationReactor") {
        problem_instances_[i].CopolymerizationReactor(N(i));

      } else {
        throw runtime_error(example + " is not a valid example.");
      }
    }
  }

  void UpdateAveragingVector(Eigen::VectorXi nave) {
    if (nave.size() != horizon_length_.size()) {
      throw std::runtime_error("nave.size() != N.size()")
    }
    num_averaging_steps_ = nave;
  }

  // runs the timing and stores the result internally
  void RunTiming() {
    int n = horizon_length_.size();

    for (int i = 0; i < n; i++) {
      // Create the controller.
      OCPGenerator& ocp = problem_instances_.at(i);
      FBstabMpc::QPData data = ocp.GetFBstabInput();
      SolverWrapper solver(data);

      // Create initial guess, for scaling tests everything is initialized at
      // the origin.
      Eigen::VectorXd z0 = Eigen::VectorXd::Zero(ocp.nz());
      Eigen::VectorXd l0 = Eigen::VectorXd::Zero(ocp.nl());
      Eigen::VectorXd v0 = Eigen::VectorXd::Zero(ocp.nv());

      Eigen::VectorXd texe(num_averaging_steps_(i));

      for (int j = 0; j < num_averaging_steps_(i); j++) {
        // Call the solver, for these tests we use the default value of x0 given
        // by the OCP generator.
        WrapperOutput out = solver.Compute(*data.x0, z0, l0, v0);
        texe(j) = out.solve_time;
        if (out.success == false) {
          throw std::runtime_error("In ScalingBenchmark::RunTiming: Solver " +
                                   solver.SolverName() + " failed.");
        }
      }

      // Compute execution time statistics.
      t_average_(i) = texe.mean();
      t_max_(i) = texe.maxCoeff();

      t_std_(i) = 0.0;
      for (int j = 0; j < num_averaging_steps_(i)) {
        t_std_(i) += pow(texe(j) - t_average_(i), 2);
      }
      t_std_(i) /= (num_averaging_steps_(i) - 1);
      t_std_(i) = sqrt(t_std_(i));
    }
  }

  /**
   * Create a vector of (approximately) logarithmically spaced vectors on the
   * interval [10^a,10^b]. E.g., logspace(0,3,4) = [1,10,100,1000].
   *
   * @param[in]  a
   * @param[in]  b
   * @param[in]  n number of points
   * @return   logspace vector
   */
  static Eigen::VectorXi logspace(double a, double b, int n) {
    // Create a logspaced vector of doubles.
    Eigen::VectorXd l = Eigen::VectorXd::Linspaced(n, a, b);
    l = l.array().exp();

    // Round to the nearest integer.
    Eigen::VectorXi out = Eigen::VectorXi::Zero(l.size());
    for (int i = 0; i < l.size(); i++) {
      out(i) = static_cast<int>(ceil(l(i)));
    }
    return out;
  }

 private:
  std::vector<OCPGenerator> problem_instances_;
  Eigen::VectorXi num_averaging_steps_;
  Eigen::VectorXi horizon_length_;

  // Average, max and standard deviations.
  Eigen::VectorXd t_average_;
  Eigen::VectorXd t_max_;
  Eigen::VectorXd t_std_;
};

}  // namespace test
}  // namespace fbstab
}  // namespace solvers
}  // namespace drake