#include "drake/solvers/fbstab/fbstab_mpc.h"

#include <cmath>
#include <gtest/gtest.h>

#include "drake/solvers/fbstab/linalg/matrix_sequence.h"
#include "drake/solvers/fbstab/linalg/static_matrix.h"
#include "drake/solvers/fbstab/test/test_helpers.h"

namespace drake {
namespace solvers {
namespace fbstab {
namespace test {

GTEST_TEST(FBstabMPC, DoubleIntegrator) {
  // set up the QP
  int N = 2;
  int nx = 2;
  int nu = 1;
  int nc = 6;

  double Q[] = {2, 0, 0, 1};
  double S[] = {1, 0};
  double R[] = {3};
  double q[] = {-2, 0};
  double r[] = {0};

  double A[] = {1, 0, 1, 1};
  double B[] = {0, 1};
  double c[] = {0, 0};
  double x0[] = {0, 0};

  double E[] = {-1, 0, 1, 0, 0, 0, 0, -1, 0, 1, 0, 0};
  double L[] = {0, 0, 0, 0, -1, 1};
  double d[] = {0, 0, -2, -2, -1, -1};

  double** Qt = test::repmat(Q, 2, 2, N + 1);
  double** Rt = test::repmat(R, 1, 1, N + 1);
  double** St = test::repmat(S, 1, 2, N + 1);
  double** qt = test::repmat(q, 2, 1, N + 1);
  double** rt = test::repmat(r, 1, 1, N + 1);

  double** At = test::repmat(A, 2, 2, N);
  double** Bt = test::repmat(B, 2, 1, N);
  double** ct = test::repmat(c, 2, 1, N);

  double** Et = test::repmat(E, 6, 2, N + 1);
  double** Lt = test::repmat(L, 6, 1, N + 1);
  double** dt = test::repmat(d, 6, 1, N + 1);

  int nz = (N + 1) * (nx + nu);
  int nl = (N + 1) * nx;
  int nv = (N + 1) * nc;

  double* z = new double[nz];
  double* l = new double[nl];
  double* v = new double[nv];
  double* y = new double[nv];

  QPDataMPC data;
  data.Q = Qt;
  data.R = Rt;
  data.S = St;
  data.q = qt;
  data.r = rt;
  data.A = At;
  data.B = Bt;
  data.c = ct;
  data.x0 = x0;

  data.E = Et;
  data.L = Lt;
  data.d = dt;

  FBstabMPC solver(N, nx, nu, nc);

  solver.UpdateOption("abs_tol", 1e-6);
  solver.SetDisplayLevel(FBstabAlgoMPC::OFF);
  SolverOut out = solver.Solve(data, z, l, v, y);

  ASSERT_EQ(out.eflag, SUCCESS);
  ASSERT_LE(out.residual, 1e-6);

  double zopt[] = {
      -5.31028204670497e-14, 5.02854354118183e-13, 0.311688311338095,
      5.35637944798588e-13,  0.311688311339015,    -0.0779220779990502,
      0.311688311339667,     0.233766233340057,    -0.103896103779874};

  double lopt[] = {-5.24675324688535,  -4.49350649223710, -3.55844155822323,
                   -0.935064934014372, -1.48051948022526, 0.233766233996585};

  double vopt[] = {1.06213597221667e-13,  -1.41190425869539e-21, 0, 0, 0, 0,
                   -1.50393600622818e-21, -8.75144622575045e-10, 0, 0, 0, 0,
                   -8.75144611157041e-10, -6.56358459377444e-10, 0, 0, 0, 0};

  for (int i = 0; i < nz; i++) {
    EXPECT_NEAR(z[i], zopt[i], 1e-8);
  }

  for (int i = 0; i < nl; i++) {
    EXPECT_NEAR(l[i], lopt[i], 1e-8);
  }

  for (int i = 0; i < nv; i++) {
    EXPECT_NEAR(v[i], vopt[i], 1e-8);
  }

  delete[] z;
  delete[] l;
  delete[] v;
  delete[] y;

  test::free_repmat(Qt, N + 1);
  test::free_repmat(Rt, N + 1);
  test::free_repmat(St, N + 1);
  test::free_repmat(qt, N + 1);
  test::free_repmat(rt, N + 1);

  test::free_repmat(At, N);
  test::free_repmat(Bt, N);
  test::free_repmat(ct, N);

  test::free_repmat(Et, N + 1);
  test::free_repmat(Lt, N + 1);
  test::free_repmat(dt, N + 1);
}

GTEST_TEST(FBstabMPC, DoubleIntegratorLongHorizon) {
  // set up the QP
  int N = 20;
  int nx = 2;
  int nu = 1;
  int nc = 6;

  double Q[] = {2, 0, 0, 1};
  double S[] = {1, 0};
  double R[] = {3};
  double q[] = {-2, 0};
  double r[] = {0};

  double A[] = {1, 0, 1, 1};
  double B[] = {0, 1};
  double c[] = {0, 0};
  double x0[] = {0, 0};

  double E[] = {-1, 0, 1, 0, 0, 0, 0, -1, 0, 1, 0, 0};
  double L[] = {0, 0, 0, 0, -1, 1};
  double d[] = {0, 0, -2, -2, -1, -1};

  double** Qt = test::repmat(Q, 2, 2, N + 1);
  double** Rt = test::repmat(R, 1, 1, N + 1);
  double** St = test::repmat(S, 1, 2, N + 1);
  double** qt = test::repmat(q, 2, 1, N + 1);
  double** rt = test::repmat(r, 1, 1, N + 1);

  double** At = test::repmat(A, 2, 2, N);
  double** Bt = test::repmat(B, 2, 1, N);
  double** ct = test::repmat(c, 2, 1, N);

  double** Et = test::repmat(E, 6, 2, N + 1);
  double** Lt = test::repmat(L, 6, 1, N + 1);
  double** dt = test::repmat(d, 6, 1, N + 1);

  int nz = (N + 1) * (nx + nu);
  int nl = (N + 1) * nx;
  int nv = (N + 1) * nc;

  double* z = new double[nz];
  double* l = new double[nl];
  double* v = new double[nv];
  double* y = new double[nv];

  QPDataMPC data;
  data.Q = Qt;
  data.R = Rt;
  data.S = St;
  data.q = qt;
  data.r = rt;
  data.A = At;
  data.B = Bt;
  data.c = ct;
  data.x0 = x0;

  data.E = Et;
  data.L = Lt;
  data.d = dt;

  FBstabMPC solver(N, nx, nu, nc);

  solver.UpdateOption("abs_tol", 1e-6);
  solver.SetDisplayLevel(FBstabAlgoMPC::OFF);
  SolverOut out = solver.Solve(data, z, l, v, y);

  ASSERT_EQ(out.eflag, SUCCESS);
  ASSERT_LE(out.residual, 1e-6);

  delete[] z;
  delete[] l;
  delete[] v;
  delete[] y;

  test::free_repmat(Qt, N + 1);
  test::free_repmat(Rt, N + 1);
  test::free_repmat(St, N + 1);
  test::free_repmat(qt, N + 1);
  test::free_repmat(rt, N + 1);

  test::free_repmat(At, N);
  test::free_repmat(Bt, N);
  test::free_repmat(ct, N);

  test::free_repmat(Et, N + 1);
  test::free_repmat(Lt, N + 1);
  test::free_repmat(dt, N + 1);
}

}  // namespace test
}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
