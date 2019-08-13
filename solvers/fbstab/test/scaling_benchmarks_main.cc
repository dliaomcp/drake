#include "drake/solvers/fbstab/test/scaling_benchmark.h"

#include <iostream>
#include <string>

#include <Eigen/Dense>

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

  drake::solvers::MathematicalProgram mp;
  auto x = mp.NewContinuousVariables(10, "x");
  auto u = mp.NewContinuousVariables(10, "u");

  // std::cout << x << std::endl;

  auto x1 = x.segment(0, 3);
  auto xsym = x1.cast<drake::symbolic::Expression>();

  auto x2 = u.segment(0, 3);

  drake::solvers::VectorXDecisionVariable z(6);
  z << x1, x2;
  std::cout << z << std::endl;

  MatrixXd Q = MatrixXd::Ones(3, 3);
  VectorXd d = VectorXd::Zero(3);

  auto q1 = mp.AddQuadraticCost(Q, d, x1);

  // std::cout << q1.evaluator()->Q() << std::endl;
  // std::cout << q1.variables() << std::endl;

  MatrixXd A = MatrixXd::Identity(3, 3);

  auto tt = x1.cast<expr>() - A.cast<expr>() * x2.cast<expr>();
  std::cout << tt << std::endl;

  auto l1 = mp.AddLinearEqualityConstraint(tt, d);
  std::cout << l1.evaluator()->A() << std::endl;

  MatrixXd E = MatrixXd::Ones(4, 3);
  MatrixXd L = -MatrixXd::Ones(4, 3);

  MatrixXd G(4, 6);
  VectorXd p = VectorXd::Ones(4) * 7;
  G << E, L;
  // std::cout << G << std::endl;

  auto lin = mp.AddLinearConstraint((G * z).array() <= p.array());
  std::cout << lin.evaluator()->A() << std::endl;
  std::cout << lin.evaluator()->upper_bound() << std::endl;
  std::cout << lin.evaluator()->lower_bound() << std::endl;

  return 0;
}
