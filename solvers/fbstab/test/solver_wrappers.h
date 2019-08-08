#pragma once

namespace drake {
namespace solvers {
namespace fbstab {
namespace test {

// All of these have to have
// An constructor and destructor and
// A compute method that returns timing information, residual information,
// a primal-dual solution, and a control input.

// Wraps FBstab
class FBstabWrapper {};

// Wraps qpOASES
class QpOasesWrapper {};

// Wraps ECOS
class EcosWrapper {};

// Wraps MathematicalProgram
// which can call OSQP, MOSEK, and Gurobi.
class MathProgramWrapper {};

}  // namespace test
}  // namespace fbstab
}  // namespace solvers
}  // namespace drake