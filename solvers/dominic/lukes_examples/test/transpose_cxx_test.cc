#include "drake/solvers/dominic/transpose_cxx.h"

#include <gtest/gtest.h>

namespace drake {
namespace solvers {
namespace dominic {
namespace test {

GTEST_TEST(TestFoo, ReturnValue) {
  int A[] = {1, 2, 3,
             4, 5, 6,
             7, 8, 9};
  int A_transpose[] = {1, 4, 7,
                       2, 5, 8,
                       3, 6, 9};
  drake::solvers::dominic::transpose(A, 3);
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
