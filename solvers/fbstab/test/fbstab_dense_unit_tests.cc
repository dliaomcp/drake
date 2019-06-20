#include "drake/solvers/fbstab/fbstab_dense.h"

#include <cmath>
#include <gtest/gtest.h>
#include <Eigen/Dense>

namespace drake {
namespace solvers {
namespace fbstab {
namespace test {

using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;

GTEST_TEST(FBstabDense, FeasibleQP) {
	MatrixXd H(2,2);
	MatrixXd A(2,2);
	VectorXd f(2);
	VectorXd b(2);

	H << 3,1,1,1;
	f << 10,5;
	A << -1,0,
		 0,1;
	b << 0,0;

	int n = f.size();
	int q = b.size();
	
	DenseQPData data;
	data.H = &H;
	data.f = &f;
	data.A = &A;
	data.b = &b;

	VectorXd z0 = Eigen::VectorXd::Zero(n);
	VectorXd v0 = Eigen::VectorXd::Zero(q);
	VectorXd y0 = Eigen::VectorXd::Zero(q);

	DenseQPVariable x0;
	x0.z = &z0;
	x0.v = &v0;
	x0.y = &y0;

	FBstabDense solver(n,q);
	solver.UpdateOption("abs_tol",1e-8);
	solver.SetDisplayLevel(FBstabAlgoDense::OFF);
	SolverOut out = solver.Solve(data,x0);

	ASSERT_EQ(out.eflag,SUCCESS);

	VectorXd zopt(2);
	VectorXd vopt(2);
	zopt << 0,-5;
	vopt << 5,0;
	for(int i =0;i<n;i++){
		EXPECT_NEAR(z0(i),zopt(i),1e-8);
	}

	for(int i =0;i<q;i++){
		EXPECT_NEAR(v0(i),vopt(i),1e-8);
	}

}

GTEST_TEST(FBstabDense, DegenerateQP) {
	MatrixXd H(2,2);
	MatrixXd A(5,2);
	VectorXd f(2);
	VectorXd b(5);

	H << 1,0,0,0;
	f << 1,0;

	A << 0,0,
		 1,0,
		 0,1,
		 -1,0,
		 0,-1;

	b << 0,3,3,-1,-1;

	int n = f.size();
	int q = b.size();
	
	DenseQPData data;
	data.H = &H;
	data.f = &f;
	data.A = &A;
	data.b = &b;

	VectorXd z0 = Eigen::VectorXd::Zero(n);
	VectorXd v0 = Eigen::VectorXd::Zero(q);
	VectorXd y0 = Eigen::VectorXd::Zero(q);

	DenseQPVariable x0;
	x0.z = &z0;
	x0.v = &v0;
	x0.y = &y0;

	FBstabDense solver(n,q);
	solver.UpdateOption("abs_tol",1e-8);
	solver.SetDisplayLevel(FBstabAlgoDense::OFF);
	SolverOut out = solver.Solve(data,x0);

	ASSERT_EQ(out.eflag,SUCCESS);

	VectorXd zopt(n);
	zopt << 1,1;
	VectorXd vopt(q);
	vopt << 0,0,0,2,0;

	for(int i =0;i<n;i++){
		EXPECT_NEAR(z0(i),zopt(i),1e-8);
	}

	for(int i =0;i<q;i++){
		EXPECT_NEAR(v0(i),vopt(i),1e-8);
	}

}

GTEST_TEST(FBstabDense, InfeasibleQP) {
	MatrixXd H(2,2);
	MatrixXd A(5,2);
	VectorXd f(2);
	VectorXd b(5);

	H << 1,0,0,0;
	f << 1,-1;

	A << 1,1,
		 1,0,
		 0,1,
		 -1,0,
		 0,-1;

	b << 0,3,3,-1,-1;

	int n = f.size();
	int q = b.size();
	
	DenseQPData data;
	data.H = &H;
	data.f = &f;
	data.A = &A;
	data.b = &b;

	VectorXd z0 = Eigen::VectorXd::Zero(n);
	VectorXd v0 = Eigen::VectorXd::Zero(q);
	VectorXd y0 = Eigen::VectorXd::Zero(q);

	DenseQPVariable x0;
	x0.z = &z0;
	x0.v = &v0;
	x0.y = &y0;

	FBstabDense solver(n,q);
	solver.UpdateOption("abs_tol",1e-8);
	solver.SetDisplayLevel(FBstabAlgoDense::OFF);
	SolverOut out = solver.Solve(data,x0);

	ASSERT_EQ(out.eflag,INFEASIBLE);

}

GTEST_TEST(FBstabDense, UnboundedQP) {
	MatrixXd H(2,2);
	MatrixXd A(4,2);
	VectorXd f(2);
	VectorXd b(4);

	H << 1,0,0,0;
	f << 1,-1;

	A << 0,0,
		 1,0,
		 -1,0,
		 0,-1;

	b << 0,3,-1,-1;

	int n = f.size();
	int q = b.size();
	
	DenseQPData data;
	data.H = &H;
	data.f = &f;
	data.A = &A;
	data.b = &b;

	VectorXd z0 = Eigen::VectorXd::Zero(n);
	VectorXd v0 = Eigen::VectorXd::Zero(q);
	VectorXd y0 = Eigen::VectorXd::Zero(q);

	DenseQPVariable x0;
	x0.z = &z0;
	x0.v = &v0;
	x0.y = &y0;

	FBstabDense solver(n,q);
	solver.UpdateOption("abs_tol",1e-8);
	solver.SetDisplayLevel(FBstabAlgoDense::OFF);
	SolverOut out = solver.Solve(data,x0);

	ASSERT_EQ(out.eflag,UNBOUNDED_BELOW);
}


}  // namespace test
}  // namespace dominic
}  // namespace solvers
}  // namespace drake
