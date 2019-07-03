#include "drake/solvers/fbstab/components/mpc_data.h"
#include "drake/solvers/fbstab/components/mpc_residual.h"
#include "drake/solvers/fbstab/components/mpc_variable.h"
#include "drake/solvers/fbstab/components/ricatti_linear_solver.h"

#include <cmath>
#include <gtest/gtest.h>

#include "drake/solvers/fbstab/linalg/matrix_sequence.h"
#include "drake/solvers/fbstab/linalg/static_matrix.h"
#include "drake/solvers/fbstab/test/test_helpers.h"

namespace drake {
namespace solvers {
namespace fbstab {
namespace test {

GTEST_TEST(MPCComponents, MPCData) {
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

  QPsizeMPC size = {N, nx, nu, nc};
  // data object
  MPCData data(Qt, Rt, St, qt, rt, At, Bt, ct, Et, Lt, dt, x0, size);

  int nz = (N + 1) * (nx + nu);
  int nl = (N + 1) * nx;
  int nv = (N + 1) * nc;

  // create inputs
  double* zm = new double[nz];
  double* lm = new double[nl];
  double* vm = new double[nv];
  StaticMatrix z(zm, nz);
  StaticMatrix l(lm, nl);
  StaticMatrix v(vm, nv);
  for (int i = 0; i < z.len(); i++) z(i) = i + 1;
  for (int i = 0; i < l.len(); i++) l(i) = i + 1;
  for (int i = 0; i < v.len(); i++) v(i) = i + 1;

  // create outputs
  double* y1m = new double[nz];
  double* y2m = new double[nl];
  double* y3m = new double[nv];
  double* y4m = new double[nz];
  double* y5m = new double[nz];
  StaticMatrix y1(y1m, nz);
  y1.fill(0);
  StaticMatrix y2(y2m, nl);
  y2.fill(0);
  StaticMatrix y3(y3m, nv);
  y3.fill(0);
  StaticMatrix y4(y4m, nz);
  y4.fill(0);
  StaticMatrix y5(y5m, nz);
  y5.fill(0);

  // test multiplications
  data.gemvH(z, 1.0, 0.0, &y1);   // y <- H*z
  data.gemvG(z, 1.0, 0.0, &y2);   // y <- G*z
  data.gemvA(z, 1.0, 0.0, &y3);   // y <- A*z
  data.gemvGT(l, 1.0, 0.0, &y4);  // y <- G'*l
  data.gemvAT(v, 1.0, 0.0, &y5);  // y <- A'*v

  // computed in matlab then hard coded
  double y1e[] = {5, 2, 10, 14, 5, 22, 23, 8, 34};
  double y2e[] = {-1, -2, -1, 0, 2, 3};
  double y3e[] = {-1, -2, 1, 2,  -3, 3, -4, -5, 4,
                  5,  -6, 6, -7, -8, 7, 8,  -9, 9};
  double y4e[] = {2, 5, 4, 2, 7, 6, -5, -6, 0};
  double y5e[] = {2, 2, 1, 2, 2, 1, 2, 2, 1};

  for (int i = 0; i < z.len(); i++) {
    ASSERT_EQ(y1(i), y1e[i]);
    ASSERT_EQ(y4(i), y4e[i]);
    ASSERT_EQ(y5(i), y5e[i]);
  }
  for (int i = 0; i < l.len(); i++) {
    ASSERT_EQ(y2(i), y2e[i]);
  }
  for (int i = 0; i < v.len(); i++) {
    ASSERT_EQ(y3(i), y3e[i]);
  }

  // test additions
  double y6e[] = {1, 2, 10, 10, 5, 22, 19, 8, 34};
  double y7e[] = {-1, -2, -1, 0, 2, 3};
  double y8e[] = {-1, -2, 5, 6,  -1, 5,  -4, -5, 8,
                  9,  -4, 8, -7, -8, 11, 12, -7, 11};
  data.axpyf(2.0, &y1);
  data.axpyh(2.0, &y2);
  data.axpyb(2.0, &y3);

  for (int i = 0; i < z.len(); i++) {
    ASSERT_EQ(y1(i), y6e[i]);
  }
  for (int i = 0; i < l.len(); i++) {
    ASSERT_EQ(y2(i), y7e[i]);
  }
  for (int i = 0; i < v.len(); i++) {
    ASSERT_EQ(y3(i), y8e[i]);
  }

  // cleanup
  delete[] zm;
  delete[] lm;
  delete[] vm;
  delete[] y1m;
  delete[] y2m;
  delete[] y3m;
  delete[] y4m;
  delete[] y5m;

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

GTEST_TEST(MPCComponents, MPCVariable) {
  // set up the QP
  int N = 2;
  int nx = 2;
  int nu = 1;
  int nc = 6;

  // int nz = (N+1)*(nx+nu);
  // int nl = (N+1)*nx;
  // int nv = (N+1)*nc;

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

  QPsizeMPC size = {N, nx, nu, nc};
  // data object
  MPCData data(Qt, Rt, St, qt, rt, At, Bt, ct, Et, Lt, dt, x0, size);

  MPCVariable x(size);
  MPCVariable y(size);

  x.LinkData(&data);
  y.LinkData(&data);

  x.Fill(1.0);
  y.Fill(1.0);

  x.axpy(y, -2.0);

  double zexp[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
  double lexp[] = {-1, -1, -1, -1, -1, -1};
  double vexp[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1,
                   -1, -1, -1, -1, -1, -1, -1, -1, -1};
  double yexp[] = {-1, -1, 3, 3, 0, 2, -1, -1, 3, 3, 0, 2, -1, -1, 3, 3, 0, 2};

  for (int i = 0; i < x.z().len(); i++) {
    ASSERT_EQ(x.z()(i), zexp[i]);
  }
  for (int i = 0; i < x.l().len(); i++) {
    ASSERT_EQ(x.l()(i), lexp[i]);
  }
  for (int i = 0; i < x.v().len(); i++) {
    ASSERT_EQ(x.v()(i), vexp[i]);
  }
  for (int i = 0; i < x.y().len(); i++) {
    ASSERT_EQ(x.y()(i), yexp[i]);
  }

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

GTEST_TEST(MPCComponents, MPCResidual) {
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

  QPsizeMPC size = {N, nx, nu, nc};
  // data object
  MPCData data(Qt, Rt, St, qt, rt, At, Bt, ct, Et, Lt, dt, x0, size);

  MPCVariable x(size);
  MPCVariable y(size);

  x.LinkData(&data);
  y.LinkData(&data);

  x.Fill(2.0);
  y.Fill(-2.0);

  double sigma = 1.0;

  MPCResidual res(size);
  res.LinkData(&data);
  res.InnerResidual(x, y, sigma);

  double rz[] = {8, 8, 14, 8, 8, 14, 6, 4, 12};
  double rl[] = {6, 6, 2, 2, 2, 2};
  double rv[] = {2.19167244568008, 2.19167244568008, 1.85147084275040,
                 1.85147084275040, 2.33389560518351, 1.62472628830921,
                 2.19167244568008, 2.19167244568008, 1.85147084275040,
                 1.85147084275040, 2.33389560518351, 1.62472628830921,
                 2.19167244568008, 2.19167244568008, 1.85147084275040,
                 1.85147084275040, 2.33389560518351, 1.62472628830921};

  for (int i = 0; i < res.z().len(); i++) {
    EXPECT_NEAR(res.z()(i), rz[i], 1e-14);
  }
  for (int i = 0; i < res.l().len(); i++) {
    EXPECT_NEAR(res.l()(i), rl[i], 1e-14);
  }
  for (int i = 0; i < res.v().len(); i++) {
    EXPECT_NEAR(res.v()(i), rv[i], 1e-14);
  }

  EXPECT_NEAR(res.Norm(), 47.0143886004911, 1e-12);

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

GTEST_TEST(MPCComponents, RicattiLinearSolver) {
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

  QPsizeMPC size = {N, nx, nu, nc};
  // data object
  MPCData data(Qt, Rt, St, qt, rt, At, Bt, ct, Et, Lt, dt, x0, size);

  MPCVariable x(size);
  MPCVariable y(size);
  x.LinkData(&data);
  y.LinkData(&data);

  x.z().fill(1);
  x.l().fill(2);
  x.v().fill(4);

  y.z().fill(2);
  y.l().fill(1);
  y.v().fill(3);

  x.InitializeConstraintMargin();
  y.InitializeConstraintMargin();

  double sigma = 1.0;

  // create the solver and factor
  RicattiLinearSolver ls(size);
  ls.LinkData(&data);
  ls.Factor(x, y, sigma);

  // create the residual then solve
  MPCResidual res(size);
  res.LinkData(&data);
  res.InnerResidual(x, y, sigma);

  MPCVariable dx(size);
  ls.Solve(res, &dx);

  // expected dx values
  // computed by the reference matlab implementation
  double dz[] = {0.0535976427784205, 0.467700500428916,  0.986932897633297,
                 -0.193383484498563, 0.305367761214063,  0.978878432270103,
                 -0.507027982371936, -0.156718234496897, 0.934679032375019};

  double dl[] = {1.94640235722158, 1.53229949957108,  0.714681627705900,
                 1.14926563684815, 0.619012259087436, 1.44096442798106};

  double dv[] = {1.95863828506912, 1.63416011976308, 2.04263318861624,
                 2.36711135392228, 2.09519447757901, 1.94459960741925,
                 2.15216504192066, 1.76135902316692, 1.84910643176471,
                 2.23991245051844, 2.10027229269928, 1.93717762013637,
                 2.39792714992857, 2.12343528835525, 1.60334432375680,
                 1.87783618533011, 2.12813713176358, 1.89644898463544};

  double dy[] = {0.0535976427784205, 0.467700500428916,  1.94640235722158,
                 1.53229949957108,   1.98693289763330,   0.0130671023667034,
                 -0.193383484498563, 0.305367761214063,  2.19338348449856,
                 1.69463223878594,   1.97887843227010,   0.0211215677298969,
                 -0.507027982371936, -0.156718234496897, 2.50702798237194,
                 2.15671823449690,   1.93467903237502,   0.0653209676249814};

  for (int i = 0; i < dx.z().len(); i++) {
    EXPECT_NEAR(dx.z()(i), dz[i], 1e-12);
  }
  for (int i = 0; i < dx.l().len(); i++) {
    EXPECT_NEAR(dx.l()(i), dl[i], 1e-12);
  }
  for (int i = 0; i < dx.v().len(); i++) {
    EXPECT_NEAR(dx.v()(i), dv[i], 1e-12);
    EXPECT_NEAR(dx.y()(i), dy[i], 1e-12);
  }

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
