#include "drake/solvers/fbstab/test/ocp_generator.h"

#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include "drake/common/drake_copyable.h"
#include "drake/solvers/fbstab/fbstab_mpc.h"

namespace drake {
namespace solvers {
namespace fbstab {
namespace test {

using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;

OCPGenerator::OCPGenerator() {}

// Returns a structure ready to be fed into FBstab.
QPDataMPC OCPGenerator::GetFBstabInput() const {
  if (!data_populated_) {
    throw std::runtime_error(
        "In OCPGenerator::GetFBstabInput: Call a problem creator method "
        "first.");
  }

  QPDataMPC s;

  s.Q = &Q_;
  s.R = &R_;
  s.S = &S_;
  s.q = &q_;
  s.r = &r_;
  s.A = &A_;
  s.B = &B_;
  s.c = &c_;
  s.E = &E_;
  s.L = &L_;
  s.d = &d_;
  s.x0 = &x0_;

  return s;
}

// void SpacecraftRelativeMotion(int N)
// void Copolymerization(int N)
// void ServoMotor(int N)

void OCPGenerator::ServoMotor(int N) {
  // Model parameters.
  const double kt = 10.0;
  const double bl = 25.0;
  const double Jm = 0.5;
  const double bm = 0.1;
  const double ktheta = 1280.2;
  const double RR = 20.0;
  const double rho = 20.0;
  const double Jl = 20 * Jm;

  const double umax = 220.0;
  const double ymax = 78.5358;

  // Continuous time matrices for
  // \dot{x} = Ax + Bu, y = Cx.
  MatrixXd A(4, 4);
  MatrixXd B(4, 1);
  A << 0, 1, 0, 0, -ktheta / Jl, -bl / Jl, ktheta / (rho * Jl), 0, 0, 0, 0, 1,
      ktheta / (rho * Jm), 0, -ktheta / (rho * rho * Jm),
      -(bm + kt * kt / RR) / Jm;
  B << 0, 0, 0, kt / (RR * Jm);

  MatrixXd C(2, 4);
  C << 1, 0, 0, 0, ktheta, 0, -ktheta / rho, 0;

  // Convert to discrete time using forward Euler.
  const double ts = 0.05;
  A = (MatrixXd::Identity(4, 4) + ts * A);
  B = ts * B;

  VectorXd c = VectorXd::Zero(4);
  VectorXd x0 = VectorXd::Zero(4);

  // Cost function
  MatrixXd Q = MatrixXd::Zero(4, 4);
  MatrixXd R = MatrixXd::Zero(1, 1);
  MatrixXd S = MatrixXd::Zero(1, 4);
  Q(0, 0) = 1000;
  R(0, 0) = 1e-4;

  constexpr double pi = 3.1415926535897;
  VectorXd xtrg(4);
  xtrg << 30 * pi / 180, 0, 0, 0;
  VectorXd utrg(1);
  utrg << 0;

  VectorXd q = -Q * xtrg;
  VectorXd r = -R * utrg;

  // Constraints
  MatrixXd E(4, 4);
  MatrixXd L(4, 1);
  VectorXd d(4);

  E << C.row(1), -C.row(1), MatrixXd::Zero(2, 4);
  L << 0, 0, 1, -1;
  d << -ymax, -ymax, -umax, -umax;

  ExtendOverHorizon(Q, R, S, q, r, A, B, c, E, L, d, x0, N);
}
// Fills internal storage with data
// for a double integrator problem with horizon N.
void OCPGenerator::DoubleIntegrator(int N) {
  MatrixXd Q(2, 2);
  MatrixXd R(1, 1);
  MatrixXd S(1, 2);
  VectorXd q(2);
  VectorXd r(1);

  MatrixXd A(2, 2);
  MatrixXd B(2, 1);
  VectorXd c(2);

  MatrixXd E(6, 2);
  MatrixXd L(6, 1);
  VectorXd d(6);

  VectorXd x0(2);

  Q << 2, 0, 0, 1;
  S << 1, 0;
  R << 3;
  q << -2, 0;
  r << 0;

  A << 1, 1, 0, 1;
  B << 0, 1;
  c << 0, 0;

  E << -1, 0, 0, -1, 1, 0, 0, 1, 0, 0, 0, 0;
  L << 0, 0, 0, 0, -1, 1;
  d << 0, 0, -2, -2, -1, -1;

  x0 << 0, 0;

  ExtendOverHorizon(Q, R, S, q, r, A, B, c, E, L, d, x0, N);
}

void OCPGenerator::ExtendOverHorizon(const MatrixXd& Q, const MatrixXd& R,
                                     const MatrixXd& S, const VectorXd& q,
                                     const VectorXd& r, const MatrixXd& A,
                                     const MatrixXd& B, const VectorXd& c,
                                     const MatrixXd& E, const MatrixXd& L,
                                     const VectorXd& d, const VectorXd& x0,
                                     int N) {
  // These are indexed from 0 to N.
  for (int i = 0; i < N + 1; i++) {
    Q_.push_back(Q);
    R_.push_back(R);
    S_.push_back(S);
    q_.push_back(q);
    r_.push_back(r);
    if (i == 0) {  // don't impose constraints on x0
      MatrixXd E0 = MatrixXd::Zero(E.rows(), E.cols());
      E_.push_back(E0);
    } else {
      E_.push_back(E);
    }
    L_.push_back(L);
    d_.push_back(d);
  }

  // These are indexed from 0 to N-1.
  for (int i = 0; i < N; i++) {
    A_.push_back(A);
    B_.push_back(B);
    c_.push_back(c);
  }
  x0_ = x0;
  N_ = N;
  nx_ = Q.rows();
  nu_ = R.rows();
  nc_ = E.rows();
  nz_ = (nx_ + nu_) * (N + 1);
  nl_ = nx_ * (N + 1);
  nv_ = nc_ * (N + 1);
  data_populated_ = true;
}

Eigen::Vector4d OCPGenerator::ProblemSize() {
  Eigen::Vector4d out;
  out << N_, nx_, nu_, nc_;
  return out;
}

}  // namespace test
}  // namespace fbstab
}  // namespace solvers
}  // namespace drake