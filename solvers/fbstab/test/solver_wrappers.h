#pragma once

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>
#include <drake/solvers/gurobi_solver.h>
#include <drake/solvers/mosek_solver.h>
#include <drake/solvers/osqp_solver.h>

#include "drake/solvers/fbstab/fbstab_mpc.h"
#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace solvers {
namespace fbstab {
namespace test {

struct WrapperOutput {
  double solve_time = 0.0;
  double setup_time = 0.0;

  int major_iters = 0;
  int minor_iters = 0;

  bool success = false;
  double residual = 0.0;

  Eigen::VectorXd z;  // Decision variable.
  Eigen::VectorXd l;  // co-state
  Eigen::VectorXd v;  // inequality duals

  Eigen::VectorXd u;  // Control signal.
};

// TODO(dliaomcp@umich.edu) write an abstract wrapper class to define the
// interface.

using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;

// Builds a MathematicalProgram object from MPC problem data.
// Returns a binding object for the constraint x0 = x(0) that can be used to
// update the parameter.
Binding<LinearEqualityConstraint> BuildMathematicalProgram(
    const FBstabMpc::QPData& data, MathematicalProgram* mp) {
  // Extract size data.
  const int N = data.B->size();
  const int nx = data.B->at(0).rows();
  const int nu = data.B->at(0).cols();
  const int nc = data.E->at(0).rows();

  // Set up the mathematical program.
  auto x = mp->NewContinuousVariables((N + 1) * nx, "x");
  auto u = mp->NewContinuousVariables((N + 1) * nu, "u");

  // Add linear-quadratic cost and inequality constraints.
  MatrixXd H(nx + nu, nx + nu);
  MatrixXd G(nc, nx + nu);
  VectorXd f(nx + nu);
  VectorXd d(nc);

  for (int i = 0; i <= N; i++) {
    // Get the variables of interest.
    auto xi = x.segment(i * nx, nx);
    auto ui = u.segment(i * nu, nu);

    VectorXDecisionVariable z(nx + nu);
    z << xi, ui;

    // The cost is 0.5 x(i)'*Q*x(i) + u(i)'*S(i)*x(i) + 0.5 u(i)'*R(i)*u(i)
    // + q(i)'*x(i) + r(i)'*u(i).
    H << data.Q->at(i), data.S->at(i).transpose(), data.S->at(i), data.R->at(i);
    f << data.q->at(i), data.r->at(i);
    mp->AddQuadraticCost(H, f, z);

    // Inequalities: E(i)*x(i) + L(i)*u(i) <= d(i).
    // MathProg wants the form lb <= Ax <= ub.
    G << data.E->at(i), data.L->at(i);
    d = -data.d->at(i);

    // There is no lower bound so set lb = -inf.
    VectorXd lb = -1.0 / 0.0 * VectorXd::Ones(nc);
    mp->AddLinearConstraint(G, lb, d, z);
  }

  // Initial state constraint.
  auto x0 = x.segment(0, nx);
  Binding<LinearEqualityConstraint> x0_cstr =
      mp->AddLinearEqualityConstraint(MatrixXd::Identity(nx, nx), *data.x0, x0);

  // Dynamic constraints:
  // x(i+1) = A(i)*x(i) + B(i)*u(i) + c(i).
  for (int i = 0; i < N; i++) {
    auto xip1 = x.segment((i + 1) * nx, nx);
    auto xi = x.segment(i * nx, nx);
    auto ui = u.segment(i * nu, nu);

    auto A = data.A->at(i).cast<symbolic::Expression>();
    auto B = data.B->at(i).cast<symbolic::Expression>();
    auto temp = xip1.cast<symbolic::Expression>() -
                A * xi.cast<symbolic::Expression>() -
                B * ui.cast<symbolic::Expression>();

    mp->AddLinearEqualityConstraint(temp, data.c->at(i));
  }

  return x0_cstr;
}

/**
 * This class wraps FBstab and implements a standardized interface to enable
 * benchmarking.
 */
class FBstabWrapper {
 public:
  FBstabWrapper(FBstabMpc::QPData data) {
    // Extract size data.
    const int N = data.B->size();
    const int nx = data.B->at(0).rows();
    const int nu = data.B->at(0).cols();
    const int nc = data.E->at(0).rows();

    // Set up the object.
    solver_ = std::make_unique<FBstabMpc>(N, nx, nu, nc);
    solver_->UpdateOption("abs_tol", 1e-6);
    solver_->UpdateOption("check_feasibility", false);
    solver_->SetDisplayLevel(FBstabAlgoMpc::Display::OFF);

    // Copy the problem data.
    data_ = data;
  }

  /** Returns the name of the solver. */
  std::string SolverName() { return "fbstab"; }

  /**
   * Updates the parameter x0 using the measured state then runs the solver,
   * starting from the primal-dual initial guess x = (z,l,v).
   *
   * @param[in] xt current state measurement
   * @param[in] z
   * @param[in] l
   * @param[in] v
   */
  WrapperOutput Compute(const Eigen::VectorXd& xt, const Eigen::VectorXd& z,
                        const Eigen::VectorXd& l, const Eigen::VectorXd& v) {
    // The data is a struct of pointers, so just make x0 point to the new value.
    Eigen::VectorXd x0 = xt;
    data_.x0 = &x0;

    // Set up initial guess.
    z_ = z;
    l_ = l;
    v_ = v;
    y_ = v;  // quick way to make y the same size as v
    FBstabMpc::QPVariable x = {&z_, &l_, &v_, &y_};

    // Call the solver.
    SolverOut out = solver_->Solve(data_, &x);

    // Populate the output struct.
    WrapperOutput w;
    w.solve_time = out.solve_time;
    w.setup_time = 0.0;
    w.major_iters = out.prox_iters;
    w.minor_iters = out.newton_iters;
    if (out.eflag == ExitFlag::SUCCESS) {
      w.success = true;
    } else {
      w.success = false;
    }
    w.residual = out.residual;
    w.z = z_;
    w.l = l_;
    w.v = v_;

    // Extract the control signal u0 from the decision variables, recall that
    // z = (x0,u0,x1,u1....).
    const int nx = data_.B->at(0).rows();
    const int nu = data_.B->at(0).cols();
    w.u = z_.segment(nx, nu);

    return w;
  }

 private:
  std::unique_ptr<FBstabMpc> solver_;
  FBstabMpc::QPData data_;

  Eigen::VectorXd z_;
  Eigen::VectorXd l_;
  Eigen::VectorXd v_;
  Eigen::VectorXd y_;
};

class MathProgWrapper {
 public:
  MathProgWrapper(const FBstabMpc::QPData& data) {
    // Build a MathProg object form the MPC problem data.
    // TODO(dliaomcp@umich.ed) Record this setup time.
    auto x0_cstr = BuildMathematicalProgram(data, &mp_);

    // This binding will be used to adjust the rhs of the constraint
    // x0 = x(t).
    x0_constraint_ =
        std::make_unique<Binding<LinearEqualityConstraint>>(x0_cstr);

    data_ = data;
  }

  virtual ~MathProgWrapper(){};
  virtual std::string SolverName() = 0;

  virtual WrapperOutput Compute(const Eigen::VectorXd& xt,
                                const Eigen::VectorXd& z,
                                const Eigen::VectorXd& l,
                                const Eigen::VectorXd& v) = 0;

 protected:
  MathematicalProgram mp_;
  std::unique_ptr<Binding<LinearEqualityConstraint>> x0_constraint_;
  FBstabMpc::QPData data_;
};

class MosekWrapper : public MathProgWrapper {
 public:
  MosekWrapper(const FBstabMpc::QPData& data) : MathProgWrapper(data){};

  std::string SolverName() { return "mosek"; }

  WrapperOutput Compute(const Eigen::VectorXd& xt, const Eigen::VectorXd& z,
                        const Eigen::VectorXd& l, const Eigen::VectorXd& v) {
    // Mosek doesn't accept an initial guess so the arguments are unused.
    MosekSolver solver;
    if (!solver.available()) {
      throw std::runtime_error(SolverName() + " is not available.");
    }

    const int N = data_.B->size();
    const int nx = data_.B->at(0).rows();
    const int nu = data_.B->at(0).cols();

    // Update the rhs of the constraint equation I*x0 = x(t).
    MatrixXd I = MatrixXd::Identity(nx, nx);
    x0_constraint_->evaluator()->UpdateCoefficients(I, xt);

    // Solve.
    MathematicalProgramResult out = solver.Solve(mp_, {}, {});
    const MosekSolverDetails& details = out.get_solver_details<MosekSolver>();

    // Return output.
    WrapperOutput s;
    s.solve_time = details.optimizer_time;
    s.setup_time = 0.0;
    s.success = out.is_success();
    s.z = out.GetSolution();
    s.l = l;
    s.v = v;
    s.u = s.z.segment((N + 1) * nx, nu);

    return s;
  }
};

class OsqpWrapper : public MathProgWrapper {
 public:
  OsqpWrapper(const FBstabMpc::QPData& data) : MathProgWrapper(data){};

  std::string SolverName() { return "osqp"; }

  WrapperOutput Compute(const Eigen::VectorXd& xt, const Eigen::VectorXd& z,
                        const Eigen::VectorXd& l, const Eigen::VectorXd& v) {
    OsqpSolver solver;
    if (!solver.available()) {
      throw std::runtime_error("OSQP is not available.");
    }

    const int N = data_.B->size();
    const int nx = data_.B->at(0).rows();
    const int nu = data_.B->at(0).cols();

    // Update the rhs of the constraint equation I*x0 = x(t).
    MatrixXd I = MatrixXd::Identity(nx, nx);
    x0_constraint_->evaluator()->UpdateCoefficients(I, xt);

    // Solve.
    // TODO(dliaomcp@umich.edu)
    MathematicalProgramResult out = solver.Solve(mp_, z, {});
    const OsqpSolverDetails& details = out.get_solver_details<OsqpSolver>();

    // Return output.
    WrapperOutput s;
    s.solve_time = details.solve_time + details.polish_time;
    s.setup_time = details.setup_time;
    s.success = out.is_success();
    s.major_iters = details.iter;
    s.residual = details.dual_res > details.primal_res ? details.dual_res
                                                       : details.primal_res;
    s.z = out.GetSolution();
    s.l = l;
    s.v = v;
    s.u = s.z.segment((N + 1) * nx, nu);

    return s;
  }
};

class GurobiWrapper : public MathProgWrapper {
 public:
  GurobiWrapper(const FBstabMpc::QPData& data) : MathProgWrapper(data){};

  std::string SolverName() { return "gurobi"; }

  WrapperOutput Compute(const Eigen::VectorXd& xt, const Eigen::VectorXd& z,
                        const Eigen::VectorXd& l, const Eigen::VectorXd& v) {
    GurobiSolver solver;
    if (!solver.available()) {
      throw std::runtime_error("gurobi is not available.");
    }

    const int N = data_.B->size();
    const int nx = data_.B->at(0).rows();
    const int nu = data_.B->at(0).cols();

    // Update the rhs of the constraint equation I*x0 = x(t).
    MatrixXd I = MatrixXd::Identity(nx, nx);
    x0_constraint_->evaluator()->UpdateCoefficients(I, xt);

    // Solve.
    MathematicalProgramResult out = solver.Solve(mp_, z, {});
    const GurobiSolverDetails& details = out.get_solver_details<GurobiSolver>();

    // Return output.
    WrapperOutput s;
    s.solve_time = details.optimizer_time;
    s.success = out.is_success();
    s.z = out.GetSolution();
    s.l = l;
    s.v = v;
    s.u = s.z.segment((N + 1) * nx, nu);

    return s;
  }
};

// Wraps qpOASES
class QpOasesWrapper {};

// Wraps ECOS
class EcosWrapper {};

}  // namespace test
}  // namespace fbstab
}  // namespace solvers
}  // namespace drake