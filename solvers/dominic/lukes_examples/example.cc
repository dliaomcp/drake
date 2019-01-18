#include "drake/solvers/dominic/transpose.h"
#include "drake/solvers/dominic/transpose_cxx.h"

#include <iostream>
#include <Eigen/Dense>

int main(int argc, char **argv)
{
  (void)argc; // suppress unused variable warning/error
  (void)argv; // suppress unused variable warning/error

  // Calling a C API for the transpose but using a Eigen C++ to view results
  // size could be specified at runtime or compile time
  double A[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  Eigen::Map<Eigen::MatrixXd> A_as_eigen_matrix(A, 3, 3);
  std::cout << "A = \n" << A_as_eigen_matrix << '\n';
  transpose(A, 3);
  std::cout << "A' = \n" << A_as_eigen_matrix << '\n';

  // Calling a C++ API for the transpose; works with generic types, not just
  // double
  // size could be specified at runtime or compile time
  int B[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Eigen::Map<Eigen::MatrixXi> B_as_eigen_matrix(B, 3, 3);
  std::cout << "B = \n" << B_as_eigen_matrix << '\n';
  drake::solvers::dominic::transpose(B, 3);
  std::cout << "B' = \n" << B_as_eigen_matrix << '\n';

  return 0;
}
