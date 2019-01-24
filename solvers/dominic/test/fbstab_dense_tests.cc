#include "drake/solvers/dominic/dense_components.h"

#include <cmath>
#include <gtest/gtest.h>

#include "drake/solvers/dominic/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace dominic {
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
	
}


// GTEST_TEST(FBstabDense, DenseLinearSolver) {
	
// }



}  // namespace test
}  // namespace dominic
}  // namespace solvers
}  // namespace drake
