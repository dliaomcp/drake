#include "drake/solvers/fbstab/components/test/mpc_component_unit_tests.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace solvers {
namespace fbstab {
namespace test {

/**
 * Runs unit tests for the MPC components. See
 * mpc_component_unit_tests.h for documentation.
 */
GTEST_TEST(MPCData, GEMVH) {
  MpcComponentUnitTests test;
  test.GEMVH();
}

GTEST_TEST(MPCData, GEMVA) {
  MpcComponentUnitTests test;
  test.GEMVA();
}

GTEST_TEST(MPCData, GEMVG) {
  MpcComponentUnitTests test;
  test.GEMVG();
}

GTEST_TEST(MPCData, GEMVGT) {
  MpcComponentUnitTests test;
  test.GEMVGT();
}

GTEST_TEST(MPCData, GEMVAT) {
  MpcComponentUnitTests test;
  test.GEMVAT();
}

GTEST_TEST(MPCData, AXPYF) {
  MpcComponentUnitTests test;
  test.AXPYF();
}

GTEST_TEST(MPCData, AXPYH) {
  MpcComponentUnitTests test;
  test.AXPYH();
}

GTEST_TEST(MPCData, AXPYB) {
  MpcComponentUnitTests test;
  test.AXPYB();
}

GTEST_TEST(MPCVariable, AXPY) {
  MpcComponentUnitTests test;
  test.Variable();
}

GTEST_TEST(MPCResidual, InnerResidual) {
  MpcComponentUnitTests test;
  test.InnerResidual();
}

GTEST_TEST(MPCFeasibility, SanityCheck) {
  MpcComponentUnitTests test;
  test.FeasibilitySanityCheck();
}

GTEST_TEST(MPCLinearSolver, RicattiRecursion) {
  MpcComponentUnitTests test;
  test.RicattiRecursion();
}

}  // namespace test
}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
