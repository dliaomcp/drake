#include "drake/solvers/dominic/transpose.h"
#include <stdio.h>

void print_row_major_int_matrix(double *A, size_t N)
{
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
        printf("%f ", A[j*N + i]);
    }
    printf("\n");
  }
}



int main(int argc, char **argv)
{
  (void)argc; // suppress unused variable warning/error
  (void)argv; // suppress unused variable warning/error


  double A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  printf("A =\n");
  print_row_major_int_matrix(A, 3);
  transpose(A, 3);
  printf("A' =\n");
  print_row_major_int_matrix(A, 3);

  return 0;
}
