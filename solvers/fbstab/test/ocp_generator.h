#pragma once

#include <vector>

#include <Eigen/Dense>
#include "drake/common/drake_copyable.h"
#include "drake/solvers/fbstab/fbstab_mpc.h"

namespace drake {
namespace solvers {
namespace fbstab {
namespace test {

/**
 * This class is used to create OCPs in a format FBstab accepts.
 * It stores the problem data internally and only passes pointers
 * to FBstab. Make sure these pointers are valid for the length
 * of the solve.
 */
class OCPGenerator {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(OCPGenerator);
  OCPGenerator();

  // Returns a structure ready to be fed into FBstab.
  QPDataMPC GetFBstabInput() const;

  // void SpacecraftRelativeMotion(int N)
  // void Copolymerization(int N)
  // void ServoMotor(int N)

  // Fills internal storage with data
  // for a double integrator problem with horizon N.
  void DoubleIntegrator(int N);

  Eigen::Vector4d ProblemSize();
  int nz() const { return nz_; }
  int nl() const { return nl_; }
  int nv() const { return nv_; }

 private:
  std::vector<Eigen::MatrixXd> Q_;
  std::vector<Eigen::MatrixXd> R_;
  std::vector<Eigen::MatrixXd> S_;
  std::vector<Eigen::VectorXd> q_;
  std::vector<Eigen::VectorXd> r_;

  std::vector<Eigen::MatrixXd> A_;
  std::vector<Eigen::MatrixXd> B_;
  std::vector<Eigen::VectorXd> c_;

  std::vector<Eigen::MatrixXd> E_;
  std::vector<Eigen::MatrixXd> L_;
  std::vector<Eigen::VectorXd> d_;

  Eigen::VectorXd x0_;

  int nz_ = 0;
  int nl_ = 0;
  int nv_ = 0;
  int N_ = 0;
  int nx_ = 0;
  int nu_ = 0;
  int nc_ = 0;

  bool data_populated_ = false;
};

}  // namespace test
}  // namespace fbstab
}  // namespace solvers
}  // namespace drake