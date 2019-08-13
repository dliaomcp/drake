#include "drake/solvers/fbstab/test/solver_wrappers.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "drake/solvers/fbstab/fbstab_mpc.h"
#include "drake/solvers/fbstab/test/ocp_generator.h"

namespace drake {
namespace solvers {
namespace fbstab {
namespace test {
namespace {

GTEST_TEST(SolverWrappers, FBstab) {
  OcpGenerator ocp;
  ocp.ServoMotor();

  FBstabMpc::QPData data = ocp.GetFBstabInput();
  FBstabWrapper w(data);

  Eigen::VectorXd z0 = Eigen::VectorXd::Zero(ocp.nz());
  Eigen::VectorXd l0 = Eigen::VectorXd::Zero(ocp.nl());
  Eigen::VectorXd v0 = Eigen::VectorXd::Zero(ocp.nv());
  Eigen::VectorXd xt = *data.x0;

  WrapperOutput out = w.Compute(xt, z0, l0, v0);

  ASSERT_TRUE(out.success);
}

GTEST_TEST(SolverWrappers, Mosek) {
  OcpGenerator ocp;
  ocp.ServoMotor();

  FBstabMpc::QPData data = ocp.GetFBstabInput();
  MosekWrapper w(data);

  Eigen::VectorXd z0 = Eigen::VectorXd::Zero(ocp.nz());
  Eigen::VectorXd l0 = Eigen::VectorXd::Zero(ocp.nl());
  Eigen::VectorXd v0 = Eigen::VectorXd::Zero(ocp.nv());
  Eigen::VectorXd xt = *data.x0;

  WrapperOutput out = w.Compute(xt, z0, l0, v0);

  ASSERT_TRUE(out.success);
}

}  // namespace
}  // namespace test
}  // namespace fbstab
}  // namespace solvers
}  // namespace drake