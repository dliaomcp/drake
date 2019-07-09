#include "drake/solvers/fbstab/test/mpc_component_unit_tests.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace solvers {
namespace fbstab {
namespace test {

/**
 * @file Runs unit tests for the Dense MPC components. See
 * mpc_component_unit_tests.h for documentation.
 */
GTEST_TEST(MPCData, GEMVH) {
  MPCComponentUnitTests test;
  test.GEMVH();
}

GTEST_TEST(MPCData, GEMVA) {
  MPCComponentUnitTests test;
  test.GEMVA();
}

GTEST_TEST(MPCData, GEMVG) {
  MPCComponentUnitTests test;
  test.GEMVG();
}

GTEST_TEST(MPCData, GEMVGT) {
  MPCComponentUnitTests test;
  test.GEMVGT();
}

GTEST_TEST(MPCData, GEMVAT) {
  MPCComponentUnitTests test;
  test.GEMVAT();
}

GTEST_TEST(MPCData, AXPYF) {
  MPCComponentUnitTests test;
  test.AXPYF();
}

GTEST_TEST(MPCData, AXPYH) {
  MPCComponentUnitTests test;
  test.AXPYH();
}

GTEST_TEST(MPCData, AXPYB) {
  MPCComponentUnitTests test;
  test.AXPYB();
}

}  // namespace test
}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
