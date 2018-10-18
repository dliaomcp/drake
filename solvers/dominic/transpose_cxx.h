#pragma once
#include <cstddef> // for std::size_t

namespace drake {
namespace solvers {
namespace dominic {

template <typename T>
void transpose(T *A, std::size_t N);

}  // namespace dominic
}  // namespace solvers
}  // namespace drake

#include "drake/solvers/dominic/transpose_cxx_impl.h"
