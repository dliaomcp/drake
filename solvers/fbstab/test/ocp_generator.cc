#include "drake/solvers/fbstab/test/ocp_generator.h"

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

  // These are indexed from 0 to N.
  for (int i = 0; i < N + 1; i++) {
    Q_.push_back(Q);
    R_.push_back(R);
    S_.push_back(S);
    q_.push_back(q);
    r_.push_back(r);

    E_.push_back(E);
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
  nx_ = 2;
  nu_ = 1;
  nc_ = 6;
  nz_ = (2 + 1) * (N + 1);
  nl_ = 2 * (N + 1);
  nv_ = 6 * (N + 1);
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