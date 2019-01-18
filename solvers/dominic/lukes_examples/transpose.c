#include "drake/solvers/dominic/transpose.h"

inline size_t idx(size_t i, size_t j, size_t N)
{
  return j*N + i;
}

void transpose(double *A, size_t N)
{
  if (A != NULL) {
    for (size_t i = 0; i < N - 1; ++i) {
      for (size_t j = i + 1; j < N; ++j) {
        const double tmp = A[idx(i, j, N)];
        A[idx(i, j, N)] = A[idx(j, i, N)];
        A[idx(j, i, N)] = tmp;
      }
    }
  }
}
