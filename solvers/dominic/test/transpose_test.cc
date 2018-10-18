#include "drake/solvers/dominic/transpose.h"

#include <gtest/gtest.h>

namespace drake {
namespace solvers {
namespace dominic {
namespace test {

GTEST_TEST(TestTranspose, ReturnValue) {
  double A[] = {1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0};
  double A_transpose[] = {1.0, 4.0, 7.0,
                          2.0, 5.0, 8.0,
                          3.0, 6.0, 9.0};
  transpose(A, 3);
  for (std::size_t i = 0; i < sizeof A / sizeof A[0]; ++i) {
    ASSERT_EQ(A[i], A_transpose[i]);
  }
}

}  // namespace test
}  // namespace dominic
}  // namespace solvers
}  // namespace drake

// Not needed unless `use_default_main == False`
//int main(int argc, char **argv) {
//  ::testing::InitGoogleTest(&argc, argv);
//  return RUN_ALL_TESTS();
//}
//
