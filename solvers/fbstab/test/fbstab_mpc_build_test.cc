#include "drake/solvers/fbstab/fbstab_mpc.h"

#include <cmath>

#include "drake/solvers/fbstab/linalg/matrix_sequence.h"
#include "drake/solvers/fbstab/linalg/static_matrix.h"
#include "drake/solvers/fbstab/test/test_helpers.h"

using namespace drake::solvers::fbstab;
using namespace std;
int main() {
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

  solver.SetDisplayLevel(FBstabAlgoMPC::ITER);
  solver.Solve(data, z, l, v, y);

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
  return 0;
}
