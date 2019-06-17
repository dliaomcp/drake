#include "drake/solvers/fbstab/components/dense_data.h"
#include "drake/solvers/fbstab/components/dense_variable.h"
#include "drake/solvers/fbstab/components/dense_residual.h"
#include "drake/solvers/fbstab/components/dense_linear_solver.h"
#include "drake/solvers/fbstab/components/dense_feasibility.h"

#include <cmath>
#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace solvers {
namespace fbstab {
namespace test {


using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;

GTEST_TEST(FBstabDense, DenseVariable) {

	// create data object
	double H[] = {3,1,1,1};
	double f[] = {1,6};
	double A[] = {-1,0,0,1};
	double b[] = {0,-1};
	int n = 2;
	int q = 2;

	DenseQPsize size = {n,q};
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
		ASSERT_EQ(y.v()(i),0.0);
	}

	// constraint margin
	x.InitializeConstraintMargin();
	y.InitializeConstraintMargin();

	double resx[] = {1,-2};
	double resy[] = {-1,0};
	for(int i = 0;i<q;i++){
		EXPECT_DOUBLE_EQ(x.y()(i),resx[i]);
		EXPECT_DOUBLE_EQ(y.y()(i),resy[i]);
	}

	// axpy
	double a = 0.35;
	y.axpy(x,a);
	double res1[] = {-0.65,-0.65};
	double res2[] = {0.35,0.35};
	double res3[] = {-0.65,-0.35};
	for(int i = 0;i<n;i++){
		EXPECT_DOUBLE_EQ(y.z()(i),res1[i]);
	}

	for(int i = 0;i<q;i++){
		EXPECT_DOUBLE_EQ(y.v()(i),res2[i]);
		EXPECT_DOUBLE_EQ(y.y()(i),res3[i]);
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

	DenseQPsize size = {n,q};
	DenseData data(H,f,A,b,size);

	// create variables
	DenseVariable x(size);
	DenseVariable xbar(size);

	x.LinkData(&data);
	xbar.LinkData(&data);

	x.z()(0) = 1;   x.z()(1) = 5;
	x.v()(0) = 0.4; x.v()(1) = 2;
	x.InitializeConstraintMargin();

	xbar.z()(0) = -5; xbar.z()(1) = 6;
	xbar.v()(0) = -9; xbar.v()(1) = 1;
	xbar.InitializeConstraintMargin();

	DenseResidual r(size);
	r.LinkData(&data);

	double sigma = 0.5;

	// PFB residual calculation
	r.InnerResidual(x,xbar,sigma);

	double rz_expected[] = {11.6,13.5};
	double rv_expected[] = {0.480683041678573,-8.88473245759182};

	for(int i = 0;i<n;i++){
		EXPECT_NEAR(r.z()(i),rz_expected[i],1e-14);
	}
	for(int i =0;i<q;i++){
		EXPECT_NEAR(r.v()(i),rv_expected[i],1e-14);
	}

	// Natural residual calculation
	r.NaturalResidual(x);

	rz_expected[0] = 8.6;
	rz_expected[1] = 14.0;
	rv_expected[0] = 0.4;
	rv_expected[1] = -6;

	for(int i = 0;i<n;i++){
		EXPECT_NEAR(r.z()(i),rz_expected[i],1e-14);
	}
	for(int i =0;i<q;i++){
		EXPECT_NEAR(r.v()(i),rv_expected[i],1e-14);
	}

	// Penalized Natural Residual calculation
	r.PenalizedNaturalResidual(x);

	rz_expected[0] = 8.6;
	rz_expected[1] = 14.0;
	rv_expected[0] = 0.4;
	rv_expected[1] = -5.7;

	for(int i = 0;i<n;i++){
		EXPECT_NEAR(r.z()(i),rz_expected[i],1e-14);
	}
	for(int i =0;i<q;i++){
		EXPECT_NEAR(r.v()(i),rv_expected[i],1e-14);
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

	DenseQPsize size = {n,q};
	DenseData data(H,f,A,b,size);

	// create variables
	DenseVariable x(size);
	DenseVariable dx(size);

	x.LinkData(&data);
	dx.LinkData(&data);

	x.z()(0) = 1;   x.z()(1) = 5;
	x.v()(0) = 0.4; x.v()(1) = 2;
	x.InitializeConstraintMargin();

	dx.z()(0) = -5; dx.z()(1) = 6;
	dx.v()(0) = -9; dx.v()(1) = 1;
	dx.InitializeConstraintMargin();

	DenseResidual r(size);
	r.LinkData(&data);

	double sigma = 0.5;

	// Residual calculation
	r.PenalizedNaturalResidual(x);
	r.Negate();

	// create the linear solver object
	DenseLinearSolver solver(size);
	solver.LinkData(&data);

	// factor and solve
	solver.Factor(x,dx,sigma);
	solver.Solve(r,&dx);

	double dz_exp[] = {-0.752407327230274,-6.29141168496996};
	double dv_exp[] = {-0.324837330275923,-3.81047514531479};
	double dy_exp[] = {-0.752407327230274,5.29141168496996};

	for(int i = 0;i<n;i++){
		EXPECT_NEAR(dx.z()(i),dz_exp[i],1e-14);
	}

	for(int i = 0;i<q;i++){
		EXPECT_NEAR(dx.v()(i),dv_exp[i],1e-14);
		EXPECT_NEAR(dx.y()(i),dy_exp[i],1e-14);
	}	
}


GTEST_TEST(FBstabDense, InfeasibilityDetection) {
	// this QP is infeasible and has no solution
	// Example from https://arxiv.org/pdf/1901.04046.pdf
	double H[] = {1,0,0,0};
	double f[] = {1,-1};
	double A[] = {1,1,0,-1,0,1,0,1,0,-1};
	double b[] = {0,3,3,-1,-1};

	int n = 2;
	int q = 5;

	DenseQPsize size = {n,q};
	DenseData data(H,f,A,b,size);

	DenseVariable dx(size);
	dx.LinkData(&data);


	// this vector v = [1 0 0 1 1] is a certificate of primal infeasibility
	// Its an application of the Friedholm alternative
	dx.z()(0) = 0;   
	dx.z()(1) = 0;

	dx.v()(0) = 1;   
	dx.v()(1) = 0;
	dx.v()(2) = 0;
	dx.v()(3) = 1;
	dx.v()(4) = 1;
	dx.InitializeConstraintMargin();

	DenseFeasibility feas(size);
	feas.LinkData(&data);
	feas.ComputeFeasibility(dx,1e-8);

	ASSERT_TRUE(feas.IsDualFeasible());
	ASSERT_FALSE(feas.IsPrimalFeasible());
}

GTEST_TEST(FBstabDense, UnboundednessDetection) {
	// this QP is unbounded below and has no solution
	// Example from https://arxiv.org/pdf/1901.04046.pdf
	double H[] = {1,0,0,0};
	double f[] = {1,-1};
	double A[] = {0,1,-1,0,0,0,0,-1};
	double b[] = {0,3,-1,-1};

	int n = 2;
	int q = 4;

	DenseQPsize size = {n,q};
	DenseData data(H,f,A,b,size);

	DenseVariable dx(size);
	dx.LinkData(&data);

	// The direction x = [0 1] is a direction of unbounded descent 
	// which certifies dual infeasibility.
	dx.z()(0) = 0;   
	dx.z()(1) = 1;

	dx.v().fill(0);

	dx.InitializeConstraintMargin();

	DenseFeasibility feas(size);
	feas.LinkData(&data);
	feas.ComputeFeasibility(dx,1e-8);

	ASSERT_FALSE(feas.IsDualFeasible());
	ASSERT_TRUE(feas.IsPrimalFeasible());
}



}  // namespace test
}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
