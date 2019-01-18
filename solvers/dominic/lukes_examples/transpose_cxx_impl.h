#pragma once

namespace drake {
namespace solvers {
namespace dominic {

inline std::size_t idx(std::size_t i, std::size_t j, std::size_t N)
{
  return j*N + i;
}

template <typename T>
void transpose(T *A, std::size_t N)
{
  if (A != nullptr) {
    for (std::size_t i = 0; i < N - 1; ++i) {
      for (std::size_t j = i + 1; j < N; ++j) {
        const T tmp = A[idx(i, j, N)];
        A[idx(i, j, N)] = A[idx(j, i, N)];
        A[idx(j, i, N)] = tmp;
      }
    }
  }
}

}  // namespace dominic
}  // namespace solvers
}  // namespace drake
