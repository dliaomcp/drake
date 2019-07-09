#include "drake/solvers/fbstab/mpc_components/mpc_data.h"

#include <stdexcept>
#include <Eigen/Dense>

namespace drake {
namespace solvers {
namespace fbstab {

using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;
using Map = Eigen::Map<Eigen::MatrixXd>;

MPCData::MPCData(const std::vector<Eigen::MatrixXd>* Q,
                 const std::vector<Eigen::MatrixXd>* R,
                 const std::vector<Eigen::MatrixXd>* S,
                 const std::vector<Eigen::VectorXd>* q,
                 const std::vector<Eigen::VectorXd>* r,
                 const std::vector<Eigen::MatrixXd>* A,
                 const std::vector<Eigen::MatrixXd>* B,
                 const std::vector<Eigen::VectorXd>* c,
                 const std::vector<Eigen::MatrixXd>* E,
                 const std::vector<Eigen::MatrixXd>* L,
                 const std::vector<Eigen::VectorXd>* d,
                 const Eigen::VectorXd* x0) {
  if (Q == nullptr || R == nullptr || S == nullptr || q == nullptr ||
      r == nullptr || A == nullptr || B == nullptr || c == nullptr ||
      E == nullptr || L == nullptr || d == nullptr || x0 == nullptr) {
    throw std::runtime_error("A null poiner was passed to MPCData::MPCData.");
  }

  Q_ = Q;
  R_ = R;
  S_ = S;
  q_ = q;
  r_ = r;
  A_ = A;
  B_ = B;
  c_ = c;
  E_ = E;
  L_ = L;
  d_ = d;
  x0_ = x0;

  validate_length();
  validate_size();

  N_ = B_->size();
  nx_ = B_->at(0).rows();
  nu_ = B_->at(0).cols();
  nc_ = E_->at(0).rows();

  nz_ = (N_ + 1) * (nx_ + nu_);
  nl_ = (N_ + 1) * nx_;
  nv_ = (N_ + 1) * nc_;
}

void MPCData::gemvH(const Eigen::VectorXd& x, double a, double b,
                    Eigen::VectorXd* y) {
  if (x.size() != nz_ || y->size() != nz_) {
    throw std::runtime_error("Size mismatch in MPCData::gemvH.");
  }
  if (b == 0.0) {
    y->fill(0.0);
  } else if (b != 1.0) {
    (*y) *= b;
  }
  // Create reshaped views of input and output vectors.
  Map w(y->data(), nx_ + nu_,
        N_ + 1);  // w = reshape(y, [nx + nu, N + 1]);
  Map v(const_cast<double*>(x.data()), nx_ + nu_,
        N_ + 1);  // v = reshape(x, [nx + nu, N + 1]);

  for (int i = 0; i < N_ + 1; i++) {
    const MatrixXd& Q = Q_->at(i);
    const MatrixXd& S = S_->at(i);
    const MatrixXd& R = R_->at(i);

    // These variables alias w.
    auto yx = w.block(0, i, nx_, 1);
    auto yu = w.block(nx_, i, nu_, 1);

    // These variables alias v.
    const auto vx = v.block(0, i, nx_, 1);
    const auto vu = v.block(nx_, i, nu_, 1);

    // yx += a*(Q(i)*vx + S(i)'*vu)
    yx.noalias() += a * Q * vx;
    yx.noalias() += a * S.transpose() * vu;

    // yu += a*(S(i)*vx + R(i)*vu)
    yu.noalias() += a * S * vx;
    yu.noalias() += a * R * vu;
  }
}

void MPCData::gemvA(const Eigen::VectorXd& x, double a, double b,
                    Eigen::VectorXd* y) {
  if (x.size() != nz_ || y->size() != nv_) {
    throw std::runtime_error("Size mismatch in MPCData::gemvA.");
  }
  if (b == 0.0) {
    y->fill(0.0);
  } else if (b != 1.0) {
    (*y) *= b;
  }
  // Create reshaped views of input and output vectors.
  Map z(const_cast<double*>(x.data()), nx_ + nu_, N_ + 1);
  Map w(y->data(), nc_, N_ + 1);

  for (int i = 0; i < N_ + 1; i++) {
    const MatrixXd& E = E_->at(i);
    const MatrixXd& L = L_->at(i);

    // This aliases w.
    auto yi = w.col(i);

    // These alias z.
    const auto xi = z.block(0, i, nx_, 1);
    const auto ui = z.block(nx_, i, nu_, 1);

    // yi += a*(E*vx + L*vu)
    yi.noalias() += a * E * xi;
    yi.noalias() += a * L * ui;
  }
}

void MPCData::gemvG(const Eigen::VectorXd& x, double a, double b,
                    Eigen::VectorXd* y) {
  if (x.size() != nz_ || y->size() != nl_) {
    throw std::runtime_error("Size mismatch in MPCData::gemvG.");
  }
  if (b == 0.0) {
    y->fill(0.0);
  } else if (b != 1.0) {
    (*y) *= b;
  }
  // Create reshaped views of input and output vectors.
  Map z(const_cast<double*>(x.data()), nx_ + nu_, N_ + 1);
  Map w(y->data(), nx_, N_ + 1);

  w.col(0).noalias() += -a * z.block(0, 0, nx_, 1);

  for (int i = 1; i < N_ + 1; i++) {
    const MatrixXd& A = A_->at(i - 1);
    const MatrixXd& B = B_->at(i - 1);

    // Alias for the output at stage i.
    auto yi = w.col(i);
    // Aliases for the state and control at stage i - 1.
    const auto xm1 = z.block(0, i - 1, nx_, 1);
    const auto um1 = z.block(nx_, i - 1, nu_, 1);
    // Alias for the state at stage i.
    const auto xi = z.block(0, i, nx_, 1);

    // Perform the operation  y(i) += a*(A(i-1)*x(i-1) + B(i-1)u(i-1) - x(i)).
    yi.noalias() += a * A * xm1;
    yi.noalias() += a * B * um1;
    yi.noalias() += -a * xi;
  }
}

void MPCData::gemvGT(const Eigen::VectorXd& x, double a, double b,
                     Eigen::VectorXd* y) {
  if (x.size() != nl_ || y->size() != nz_) {
    throw std::runtime_error("Size mismatch in MPCData::gemvGT.");
  }
  if (b == 0.0) {
    y->fill(0.0);
  } else if (b != 1.0) {
    (*y) *= b;
  }

  // Create reshaped views of input and output vectors.
  Map v(const_cast<double*>(x.data()), nx_, N_ + 1);
  Map w(y->data(), nx_ + nu_, N_ + 1);

  for (int i = 0; i < N_; i++) {
    const MatrixXd& A = A_->at(i);
    const MatrixXd& B = B_->at(i);

    // Aliases for the dual variables at stage i and i+1;
    const auto vi = v.col(i);
    const auto vp1 = v.col(i + 1);

    // Aliases for the state and control at stage i.
    auto xi = w.block(0, i, nx_, 1);
    auto ui = w.block(nx_, i, nu_, 1);

    // x(i) += a*(-v(i) + A(i)' * v(i+1))
    xi.noalias() += -a * vi;
    xi.noalias() += a * A.transpose() * vp1;

    // u(i) += a*B(i)' * v(i+1)
    ui.noalias() += a * B.transpose() * vp1;
  }
  // The i = N step of the recursion.
  w.block(0, N_, nx_, 1).noalias() += -a * v.col(N_);
}

void MPCData::gemvAT(const Eigen::VectorXd& x, double a, double b,
                     Eigen::VectorXd* y) {
  if (x.size() != nv_ || y->size() != nz_) {
    throw std::runtime_error("Size mismatch in MPCData::gemvAT.");
  }
  if (b == 0.0) {
    y->fill(0.0);
  } else if (b != 1.0) {
    (*y) *= b;
  }
  // Create reshaped views of input and output vectors.
  Map v(const_cast<double*>(x.data()), nc_, N_ + 1);
  Map w(y->data(), nx_ + nu_, N_ + 1);

  for (int i = 0; i < N_ + 1; i++) {
    const MatrixXd& E = E_->at(i);
    const MatrixXd& L = L_->at(i);

    auto xi = w.block(0, i, nx_, 1);
    auto ui = w.block(nx_, i, nu_, 1);

    const auto vi = v.col(i);
    // x(i) += a*E(i)' * v(i)
    xi.noalias() += a * E.transpose() * vi;
    // u(i) += a*L(i)' * v(i)
    ui.noalias() += a * L.transpose() * vi;
  }
}

void MPCData::axpyf(double a, Eigen::VectorXd* y) {
  if (y->size() != nz_) {
    throw std::runtime_error("Size mismatch in MPCData::axpyf.");
  }
  // Create reshaped view of the input vector.
  Map w(y->data(), nx_ + nu_, N_ + 1);

  for (int i = 0; i < N_ + 1; i++) {
    auto xi = w.block(0, i, nx_, 1);
    auto ui = w.block(nx_, i, nu_, 1);

    xi.noalias() += a * q_->at(i);
    ui.noalias() += a * r_->at(i);
  }
}

void MPCData::axpyh(double a, Eigen::VectorXd* y) {
  if (y->size() != nl_) {
    throw std::runtime_error("Size mismatch in MPCData::axpyh.");
  }
  // Create reshaped view of the input vector.
  Map w(y->data(), nx_, N_ + 1);
  w.col(0) += -a * (*x0_);

  for (int i = 1; i < N_ + 1; i++) {
    w.col(i) += -a * c_->at(i - 1);
  }
}

void MPCData::axpyb(double a, Eigen::VectorXd* y) {
  if (y->size() != nv_) {
    throw std::runtime_error("Size mismatch in MPCData::axpyb.");
  }
  // Create reshaped view of the input vector.
  Map w(y->data(), nc_, N_ + 1);

  for (int i = 0; i < N_ + 1; i++) {
    w.col(i).noalias() += -a * d_->at(i);
  }
}

void MPCData::validate_length() {
  bool OK = true;

  // Uses an unsigned long here
  // for consistency with the return type of vector::size.
  unsigned long N = Q_->size();
  if (N <= 0) {
    throw std::runtime_error("Horizon length must be at least 1.");
  }

  OK = OK && N == R_->size();
  OK = OK && N == S_->size();
  OK = OK && N == q_->size();
  OK = OK && N == r_->size();
  OK = OK && (N - 1) == A_->size();
  OK = OK && (N - 1) == B_->size();
  OK = OK && (N - 1) == c_->size();
  OK = OK && N == E_->size();
  OK = OK && N == L_->size();
  OK = OK && N == d_->size();

  if (!OK) {
    throw std::runtime_error(
        "Sequence length mismatch in input data to MPCData.");
  }
}

void MPCData::validate_size() {
  int N = B_->size();

  int nx = Q_->at(0).rows();
  if (x0_->size() != nx) {
    throw std::runtime_error("Size mismatch in input data to MPCData.");
  }
  for (int i = 0; i < N + 1; i++) {
    if (Q_->at(i).rows() != nx || Q_->at(i).cols() != nx) {
      throw std::runtime_error("Size mismatch in Q input to MPCData.");
    }
    if (S_->at(i).cols() != nx) {
      throw std::runtime_error("Size mismatch in S input to MPCData.");
    }
    if (q_->at(i).size() != nx) {
      throw std::runtime_error("Size mismatch in q input to MPCData.");
    }
    if (E_->at(i).cols() != nx) {
      throw std::runtime_error("Size mismatch in E input to MPCData.");
    }
  }
  for (int i = 0; i < N; i++) {
    if (A_->at(i).rows() != nx || A_->at(i).cols() != nx) {
      throw std::runtime_error("Size mismatch in A input to MPCData.");
    }
    if (B_->at(i).rows() != nx) {
      throw std::runtime_error("Size mismatch in B input to MPCData.");
    }
    if (c_->at(i).size() != nx) {
      throw std::runtime_error("Size mismatch in c input to MPCData.");
    }
  }

  int nu = R_->at(0).rows();
  for (int i = 0; i < N + 1; i++) {
    if (R_->at(i).rows() != nu || R_->at(i).cols() != nu) {
      throw std::runtime_error("Size mismatch in R input to MPCData.");
    }
    if (S_->at(i).rows() != nu) {
      throw std::runtime_error("Size mismatch in S input to MPCData.");
    }
    if (r_->at(i).size() != nu) {
      throw std::runtime_error("Size mismatch in r input to MPCData.");
    }
    if (L_->at(i).cols() != nu) {
      throw std::runtime_error("Size mismatch in L input to MPCData.");
    }
  }
  for (int i = 0; i < N; i++) {
    if (B_->at(i).cols() != nu) {
      throw std::runtime_error("Size mismatch in B input to MPCData.");
    }
  }

  int nc = E_->at(0).rows();
  for (int i = 0; i < N + 1; i++) {
    if (E_->at(i).rows() != nc) {
      throw std::runtime_error("Size mismatch in E input to MPCData.");
    }
    if (L_->at(i).rows() != nc) {
      throw std::runtime_error("Size mismatch in L input to MPCData.");
    }
    if (d_->at(i).size() != nc) {
      throw std::runtime_error("Size mismatch in d input to MPCData.");
    }
  }
}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake