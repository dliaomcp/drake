#pragma once

#include <memory>
#include <string>

#include <Eigen/Dense>

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

class MosekWrapper {
 public:
  MosekWrapper(FBstabMpc::QPData data) {
    // Extract size data.
    const int N = data.B->size();
    const int nx = data.B->at(0).rows();
    const int nu = data.B->at(0).cols();
    const int nc = data.E->at(0).rows();

    // Set up the mathematical program.
    auto x = mp_.NewContinuousVariables((N + 1) * nx, "x");
    auto u = mp_.NewContinuousVariables((N + 1) * nu, "u");

    // Add linear-quadratic cost and inequality constraints.
    MatrixXd H(nx + nu, nx + nu);
    MatrixXd G(nc, nx + nu);
    VectorXd f(nx + nu);
    VectorXd d(nc);

    for (int i = 0; i <= N; i++) {
      // Get the variables of interest
      auto xi = x.segment(i * nx, nx);
      auto ui = u.segment(i * nu, nu);

      VectorXDecisionVariable z(nx + nu);
      z << xi, ui;

      // Cost 0.5 z'Hz + f'z
      H << data.Q->at(i), data.S->at(i).transpose(), data.S->at(i),
          data.R->at(i);
      f << data.q->at(i), data.r->at(i);

      mp_.AddQuadraticCost(H, f, z);

      // Inequalities Ex + Lu <= d
      G << data.E->at(i), data.L->at(i);
      d = -data.d->at(i);

      mp_.AddLinearConstraint((G * z).array() <= d.array());
    }

    // Initial state constraint.
    auto x0 = x.segment(0, nx);
    mp_.AddLinearEqualityConstraint(MatrixXd::Identity(nx, nx), *data.x0, x0);

    // Dynamic constraints
    for (int i = 0; i < N; i++) {
      auto xip1 = x.segment((i + 1) * nx, nx);
      auto xi = x.segment(i * nx, nx);
      auto ui = u.segment(i * nu, nu);

      auto A = data.A->at(i).cast<symbolic::Expression>();
      auto B = data.B->at(i).cast<symbolic::Expression>();
      auto temp = xip1.cast<symbolic::Expression>() -
                  A * xi.cast<symbolic::Expression>() -
                  B * ui.cast<symbolic::Expression>();

      mp_.AddLinearEqualityConstraint(temp, data.c->at(i));
    }
  }

 private:
  MathematicalProgram mp_;
};

// Wraps qpOASES
class QpOasesWrapper {};

// Wraps ECOS
class EcosWrapper {};

}  // namespace test
}  // namespace fbstab
}  // namespace solvers
}  // namespace drake