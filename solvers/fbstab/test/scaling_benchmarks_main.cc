#include "drake/solvers/fbstab/test/scaling_benchmark.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "drake/solvers/fbstab/test/ocp_generator.h"
#include "drake/solvers/fbstab/test/solver_wrappers.h"

using string = std::string;
namespace test = drake::solvers::fbstab::test;
using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;
using VectorXi = Eigen::VectorXi;

int main(void) {
  // Setup data
  VectorXi N2 = test::logspace(1, 3, 10);
  VectorXi N1(5);
  N1 << 1, 3, 5, 7, 9;

  const int n = N1.size() + N2.size();
  VectorXi N(n);
  N << N1, N2;

  VectorXi nave(n);
  nave << 100 * VectorXi::Ones(12), 25 * VectorXi::Ones(3);

  std::vector<string> examples{"DoubleIntegrator", "ServoMotor",
                               "CopolymerizationReactor",
                               "SpacecraftRelativeMotion"};

  // Loop over examples.
  for (int i = 0; i < static_cast<int>(examples.size()); i++) {
    std::cout << "\n  " + examples.at(i) << "\n"
              << "-----------------------------\n";
    // Setup each solver.
    test::ScalingBenchmark<test::FBstabWrapper> fbstab(N, examples.at(i));
    test::ScalingBenchmark<test::MosekWrapper> mosek(N, examples.at(i));
    test::ScalingBenchmark<test::GurobiWrapper> gurobi(N, examples.at(i));
    test::ScalingBenchmark<test::OsqpWrapper> osqp(N, examples.at(i));

    // update averaging vector
    fbstab.UpdateAveragingVector(nave);
    mosek.UpdateAveragingVector(nave);
    gurobi.UpdateAveragingVector(nave);
    osqp.UpdateAveragingVector(nave);

    // Run timings.
    fbstab.RunTiming();
    mosek.RunTiming();
    gurobi.RunTiming();
    osqp.RunTiming();

    // Write to file
    fbstab.WriteResultsToFile("fbstab_" + examples.at(i) + ".csv");
    mosek.WriteResultsToFile("mosek_" + examples.at(i) + ".csv");
    gurobi.WriteResultsToFile("gurobi_" + examples.at(i) + ".csv");
    osqp.WriteResultsToFile("osqp_" + examples.at(i) + ".csv");

    std::cout << "---------------------------\n";
  }

  // Loop over examples.
  for (int i = 0; i < static_cast<int>(examples.size()); i++) {
    std::cout << "\n  " + examples.at(i) << "\n"
              << "-----------------------------\n";
    // Setup each solver.
    test::ScalingBenchmark<test::FBstabWrapper> fbstab(N, examples.at(i));
    test::ScalingBenchmark<test::MosekWrapper> mosek(N, examples.at(i));
    test::ScalingBenchmark<test::GurobiWrapper> gurobi(N, examples.at(i));
    test::ScalingBenchmark<test::OsqpWrapper> osqp(N, examples.at(i));

    // update averaging vector
    fbstab.UpdateAveragingVector(nave);
    mosek.UpdateAveragingVector(nave);
    gurobi.UpdateAveragingVector(nave);
    osqp.UpdateAveragingVector(nave);

    // Run timings.
    fbstab.RunTiming(true);
    mosek.RunTiming(true);
    gurobi.RunTiming(true);
    osqp.RunTiming(true);

    // Write to file
    fbstab.WriteResultsToFile("fbstab_warm_" + examples.at(i) + ".csv");
    mosek.WriteResultsToFile("mosek_warm_" + examples.at(i) + ".csv");
    gurobi.WriteResultsToFile("gurobi_warm_" + examples.at(i) + ".csv");
    osqp.WriteResultsToFile("osqp_warm_" + examples.at(i) + ".csv");

    std::cout << "---------------------------\n";
  }

  return 0;
}
