#include "drake/solvers/dominic/dense_components.h"

#include <cmath>
#include <gtest/gtest.h>

#include "drake/solvers/dominic/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {
namespace test {

GTEST_TEST(FBstabDense, DenseData) {
	// data for a simple convex qp
	double H[] = {3,1,1,1};
	double f[] = {1,6};
	double A[] = {-1,0,0,1};
	double b[] = {0,0};
	int n = 2;
	int q = 2;

	QPsize size = {n,q};

	DenseData data(H,f,A,b,size);

	for(int i = 0;i< n*n; i++){
		ASSERT_EQ(data.H.data[i],H[i]);
	}

	for(int i = 0;i< n; i++){
		ASSERT_EQ(data.f.data[i],f[i]);
	}

	for(int i = 0;i< q*q; i++){
		ASSERT_EQ(data.A.data[i],A[i]);
	}

	for(int i = 0;i< q; i++){
		ASSERT_EQ(data.b.data[i],b[i]);
	}
}


GTEST_TEST(FBstabDense, DenseVariable) {

	// create data object
	double H[] = {3,1,1,1};
	double f[] = {1,6};
	double A[] = {-1,0,0,1};
	double b[] = {0,-1};
	int n = 2;
	int q = 2;

	QPsize size = {n,q};
	DenseData data(H,f,A,b,size);

	DenseVariable x(size);
	DenseVariable y(size);

	// link data
	x.LinkData(&data);
	y.LinkData(&data);

	// fill
	x.Fill(1.0);
	y.Fill(-1.0);

	// projection
	y.ProjectDuals();

	for(int i = 0;i<q;i++){
		ASSERT_EQ(y.v(i),0.0);
	}

	// constraint margin
	x.InitConstraintMargin();
	y.InitConstraintMargin();

	double resx[] = {1,-2};
	double resy[] = {-1,0};
	for(int i = 0;i<q;i++){
		EXPECT_DOUBLE_EQ(x.y(i),resx[i]);
		EXPECT_DOUBLE_EQ(y.y(i),resy[i]);
	}

	// axpy
	double a = 0.35;
	y.axpy(x,a);
	double res1[] = {-0.65,-0.65};
	double res2[] = {0.35,0.35};
	double res3[] = {-0.65,-0.35};
	for(int i = 0;i<n;i++){
		EXPECT_DOUBLE_EQ(y.z(i),res1[i]);
	}

	for(int i = 0;i<q;i++){
		EXPECT_DOUBLE_EQ(y.v(i),res2[i]);
		EXPECT_DOUBLE_EQ(y.y(i),res3[i]);
	}
}


GTEST_TEST(FBstabDense, DenseResidual) {
	// create data object
	double H[] = {3,1,1,1};
	double f[] = {1,6};
	double A[] = {-1,0,0,1};
	double b[] = {0,-1};
	int n = 2;
	int q = 2;

	QPsize size = {n,q};
	DenseData data(H,f,A,b,size);

	// create variables
	DenseVariable x(size);
	DenseVariable xbar(size);

	x.LinkData(&data);
	xbar.LinkData(&data);

	x.z(0) = 1; x.z(1) = 5;
	x.v(0) = 0.4; x.v(1) = 2;
	x.InitConstraintMargin();

	xbar.z(0) = -5; xbar.z(1) = 6;
	xbar.v(0) = -9; xbar.v(1) = 1;
	xbar.InitConstraintMargin();

	DenseResidual r(size);
	r.LinkData(&data);

	double sigma = 0.5;
	double alpha = 0.95;
	r.alpha = alpha;

	// PFB residual calculation
	r.FBresidual(x,xbar,sigma);

	double rz_expected[] = {11.6,13.5};
	double rv_expected[] = {0.480683041678573,-8.88473245759182};

	for(int i = 0;i<n;i++){
		EXPECT_NEAR(r.rz(i),rz_expected[i],1e-14);
	}
	for(int i =0;i<q;i++){
		EXPECT_NEAR(r.rv(i),rv_expected[i],1e-14);
	}

	// Natural residual calculation
	r.NaturalResidual(x);

	rz_expected[0] = 8.6;
	rz_expected[1] = 14.0;
	rv_expected[0] = 0.4;
	rv_expected[1] = -6;

	for(int i = 0;i<n;i++){
		EXPECT_NEAR(r.rz(i),rz_expected[i],1e-14);
	}
	for(int i =0;i<q;i++){
		EXPECT_NEAR(r.rv(i),rv_expected[i],1e-14);
	}

	// Penalized Natural Residual calculation
	r.PenalizedNaturalResidual(x);

	rz_expected[0] = 8.6;
	rz_expected[1] = 14.0;
	rv_expected[0] = 0.4;
	rv_expected[1] = -5.7;

	for(int i = 0;i<n;i++){
		EXPECT_NEAR(r.rz(i),rz_expected[i],1e-14);
	}
	for(int i =0;i<q;i++){
		EXPECT_NEAR(r.rv(i),rv_expected[i],1e-14);
	}

	// check the norm function
	EXPECT_NEAR(r.Norm(),22.144477369697360,1e-14);
}


GTEST_TEST(FBstabDense, DenseLinearSolver) {
	// create data object
	double H[] = {3,1,1,1};
	double f[] = {1,6};
	double A[] = {-1,0,0,1};
	double b[] = {0,-1};
	int n = 2;
	int q = 2;

	QPsize size = {n,q};
	DenseData data(H,f,A,b,size);

	// create variables
	DenseVariable x(size);
	DenseVariable dx(size);

	x.LinkData(&data);
	dx.LinkData(&data);

	x.z(0) = 1; x.z(1) = 5;
	x.v(0) = 0.4; x.v(1) = 2;
	x.InitConstraintMargin();

	dx.z(0) = -5; dx.z(1) = 6;
	dx.v(0) = -9; dx.v(1) = 1;
	dx.InitConstraintMargin();

	DenseResidual r(size);
	r.LinkData(&data);

	double sigma = 0.5;
	double alpha = 0.95;
	r.alpha = alpha;

	// PFB residual calculation
	r.PenalizedNaturalResidual(x);

	// create the linear solver object
	DenseLinearSolver solver(size);
	solver.LinkData(&data);

	// test factor
	solver.Factor(x,dx,sigma);

	double mus_exp[] = {1.17966217107061,1.54674596622464};
	double gamma_exp[] = {0.0223305769546052,1.84280375231402};

	for(int i = 0;i<q;i++){
		EXPECT_NEAR(solver.mus.data[i],mus_exp[i],1e-14);
		EXPECT_NEAR(solver.gamma.data[i],gamma_exp[i],1e-14);
	}

	StaticMatrix K(solver.K);

	double expected_K[] = {1.87588102960501,0.533082847055906,0,1.55152490683952};

	// K need its upper triangle zeroed out for the comparison
	// by default its left unchanged
	K.tril();
	for(int i = 0;i<n;i++){
		EXPECT_NEAR(K.data[i],expected_K[i],1e-14);
	}

	r.Negate();
	// now the solve step
	solver.Solve(r,&dx);

	double dz_exp[] = {-0.752407327230274,-6.29141168496996};
	double dv_exp[] = {-0.324837330275923,-3.81047514531479};
	double dy_exp[] = {-0.752407327230274,5.29141168496996};

	for(int i = 0;i<n;i++){
		EXPECT_NEAR(dx.z(i),dz_exp[i],1e-14);
	}

	for(int i = 0;i<q;i++){
		EXPECT_NEAR(dx.v(i),dv_exp[i],1e-14);
		EXPECT_NEAR(dx.y(i),dy_exp[i],1e-14);
	}	
}

// GTEST_TEST(FBstabDense,SolverIntegration) {
// 	double zopt_exp[] = {2.49999999664944, -8.49999998818445};
// 	double vopt_exp[] = {3.65900262042578e-09,6.31591394316009e-11};
// }



}  // namespace test
}  // namespace dominic
}  // namespace solvers
}  // namespace drake
