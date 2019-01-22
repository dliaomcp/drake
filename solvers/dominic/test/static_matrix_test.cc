#include "drake/solvers/dominic/linalg/static_matrix.h"

#include <gtest/gtest.h>

namespace drake {
namespace solvers {
namespace dominic {
namespace test {


// test assignment
GTEST_TEST(TestFoo, ReturnValue) {
	double a[] = {2,10,10,5,9,2};
	StaticMatrix A(a,3,2);

	A(0,1) = 4;
	A(2,0) = 8;

	double B[] = {2,10,8,5,9,2};
	for(int i = 0;i<A.size();i++){
		ASSERT_EQ(A.data[i],B[i]);
	}
}

// test slicing

GTEST_TEST(TestFoo, ReturnValue){
	double a[] = {2,10,10,5,9,2};
	StaticMatrix A(a,3,2);

	StaticMatrix Arow = A.row(1);
	StaticMatrix Acol = A.col(1);

	double row[] = {10,9};
	double col[] = {5,9,2};

	for(int i = 0;i<A.cols();i++){
		ASSERT_EQ(Arow(i),row[i]);
	}

	for(int i = 0;i<A.rows();i++){
		ASSERT_EQ(Acol(i),col[i]);
	}

}

// test reshaping

// test axpy

// test gemv

// test gemm

// test gram

// test cholesky



}  // namespace test
}  // namespace dominic
}  // namespace solvers
}  // namespace drake
