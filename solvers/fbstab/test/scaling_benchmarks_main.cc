#include "drake/solvers/fbstab/test/scaling_benchmark.h"

#include <iostream>
#include <string>

#include <Eigen/Dense>

#include "drake/solvers/fbstab/test/ocp_generator.h"
#include "drake/solvers/fbstab/test/solver_wrappers.h"
#include "drake/solvers/mathematical_program.h"

using string = std::string;
namespace test = drake::solvers::fbstab::test;
using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;
using expr = drake::symbolic::Expression;

int main(void) {
  // Scaling horizon.
  // Eigen::VectorXi N(4);
  // N << 2, 5, 10, 100;

  // string example = "ServoMotor";
  // test::ScalingBenchmark<test::FBstabWrapper> s(N, example);
  // s.RunTiming();

  // std::cout << s.WriteResultsToFile("test.csv");

  return 0;
}
