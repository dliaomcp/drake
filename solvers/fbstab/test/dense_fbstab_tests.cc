#include "drake/solvers/fbstab/fbstab_dense.h"

#include <cmath>
#include <gtest/gtest.h>

#include "drake/solvers/fbstab/linalg/static_matrix.h"

namespace drake {
namespace solvers {
namespace fbstab {
namespace test {


GTEST_TEST(FBstabDense, FeasibleQP) {
	double H[] = {3,1,1,1};
	double f[] = {10,5};
	double A[] = {-1,0,0,1};
	double b[] = {0,0};

	int n = 2;
	int q = 2;
	
	FBstabDense solver(n,q);

	DenseQPData data;
	data.H = H;
	data.f = f;
	data.A = A;
	data.b = b;

	double z[] = {0,0};
	double v[] = {0,0};
	double y[] = {0,0};

	solver.UpdateOption("abs_tol",1e-8);
	solver.SetDisplayLevel(FBstabAlgoDense::OFF);
	SolverOut out = solver.Solve(data,z,v,y);

	ASSERT_EQ(out.eflag,SUCCESS);

	double zopt[2] = {0,-5};
	double vopt[2] = {5,0};
	for(int i =0;i<n;i++){
		EXPECT_NEAR(z[i],zopt[i],1e-8);
	}

	for(int i =0;i<q;i++){
		EXPECT_NEAR(v[i],vopt[i],1e-8);
	}

}

GTEST_TEST(FBstabDense, DegenerateQP) {
	double H[] = {1,0,0,0};
	double f[] = {1,0};
	double A[] = {0,1,0,-1,0,0,0,1,0,-1};
	double b[] = {0,3,3,-1,-1};

	int n = 2;
	int q = 5;
	FBstabDense solver(n,q);

	DenseQPData data;
	data.H = H;
	data.f = f;
	data.A = A;
	data.b = b;

	double z[] = {0,0};
	double v[5] = {0};
	double y[5] = {0};

	solver.UpdateOption("abs_tol",1e-8);
	solver.SetDisplayLevel(FBstabAlgoDense::OFF);
	SolverOut out = solver.Solve(data,z,v,y);

	ASSERT_EQ(out.eflag,SUCCESS);

	double zopt[] = {1,1};
	double vopt[] = {0,0,0,2,0};
	for(int i =0;i<n;i++){
		EXPECT_NEAR(z[i],zopt[i],1e-8);
	}

	for(int i =0;i<q;i++){
		EXPECT_NEAR(v[i],vopt[i],1e-8);
	}

}

GTEST_TEST(FBstabDense, InfeasibleQP) {
	double H[] = {1,0,0,0};
	double f[] = {1,-1};
	double A[] = {1,1,0,-1,0,1,0,1,0,-1};
	double b[] = {0,3,3,-1,-1};

	int n = 2;
	int q = 5;
	FBstabDense solver(n,q);

	DenseQPData data;
	data.H = H;
	data.f = f;
	data.A = A;
	data.b = b;

	double z[] = {0,0};
	double v[5] = {0};
	double y[5] = {0};

	solver.UpdateOption("abs_tol",1e-8);
	solver.SetDisplayLevel(FBstabAlgoDense::OFF);
	SolverOut out = solver.Solve(data,z,v,y);

	ASSERT_EQ(out.eflag,INFEASIBLE);

}

GTEST_TEST(FBstabDense, UnboundedQP) {
	double H[] = {1,0,0,0};
	double f[] = {1,-1};
	double A[] = {0,1,-1,0,0,0,0,-1};
	double b[] = {0,3,-1,-1};

	int n = 2;
	int q = 4;
	FBstabDense solver(n,q);

	DenseQPData data;
	data.H = H;
	data.f = f;
	data.A = A;
	data.b = b;

	double z[] = {0,0};
	double v[4] = {0};
	double y[4] = {0};

	solver.UpdateOption("abs_tol",1e-8);
	solver.SetDisplayLevel(FBstabAlgoDense::OFF);
	SolverOut out = solver.Solve(data,z,v,y);

	ASSERT_EQ(out.eflag,UNBOUNDED_BELOW);
}


}  // namespace test
}  // namespace dominic
}  // namespace solvers
}  // namespace drake
