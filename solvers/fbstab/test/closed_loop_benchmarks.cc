#include "drake/solvers/fbstab/test/timing_simulator.h"

#include <iostream>
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
  int nave = 1;
  // std::vector<string> examples{"SpacecraftRelativeMotion", "ServoMotor",
  //                              "CopolymerizationReactor"};

  std::vector<string> examples{"ServoMotor"};

  for (int i = 0; i < static_cast<int>(examples.size()); i++) {
    std::cout << "\n  " + examples.at(i) << "\n"
              << "-----------------------------\n";

    test::TimingSimulator<test::FBstabWrapper> s1(examples.at(i));  // coldstart
    test::TimingSimulator<test::FBstabWrapper> s2(examples.at(i));  // warmstart
    test::TimingSimulator<test::MosekWrapper> s3(examples.at(i));
    test::TimingSimulator<test::GurobiWrapper> s4(examples.at(i));
    test::TimingSimulator<test::OsqpWrapper> s5(examples.at(i));

    s1.SetAveragingSteps(nave);
    s2.SetAveragingSteps(nave);
    s3.SetAveragingSteps(nave);
    s4.SetAveragingSteps(nave);
    s5.SetAveragingSteps(nave);

    bool warm = true;
    bool cold = false;

    s1.RunSimulation(warm);
    s2.RunSimulation(cold);
    s3.RunSimulation(cold);
    s4.RunSimulation(cold);
    // OSQP's MathProg wrapper doesn't currently
    // support warmstarting -> make this warm once it does
    s5.RunSimulation(cold);

    s1.WriteResultsToFile("cl_fbstab_warm_" + examples.at(i) + ".csv");
    s2.WriteResultsToFile("cl_fbstab_cold_" + examples.at(i) + ".csv");
    s3.WriteResultsToFile("cl_mosek_cold_" + examples.at(i) + ".csv");
    s4.WriteResultsToFile("cl_gurobi_cold_" + examples.at(i) + ".csv");
    s5.WriteResultsToFile("cl_osqp_cold_" + examples.at(i) + ".csv");

    std::cout << "---------------------------\n";
  }

  return 0;
}