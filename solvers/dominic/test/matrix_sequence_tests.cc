#include "drake/solvers/dominic/linalg/static_matrix.h"
#include "drake/solvers/dominic/linalg/matrix_sequence.h"

#include <gtest/gtest.h>


#define DOUBLE_ABS_TOL 1e-14
namespace drake {
namespace solvers {
namespace fbstab {
namespace test {


GTEST_TEST(MatrixSequenceTest, IndexTest) {
	double a[2][6] = {{1,2,3,4,5,6}, {3,4,5,6,7,8}};
	double *a1[2];
	a1[0] = &a[0][0];
	a1[1] = &a[1][0];

	MatrixSequence A(a1,2,2,3);

	StaticMatrix A0 = A(0);
	StaticMatrix A1 = A(1);

	for(int i = 0;i<A0.size();i++){
		ASSERT_EQ(A0.data[i],a[0][i]);
	}

	for(int i = 0;i<A1.size();i++){
		ASSERT_EQ(A1.data[i],a[1][i]);
	}
}


GTEST_TEST(MatrixSequenceTest, IndexTest2) {
	double** a = new double*[2];
	for(int i = 0;i<2;i++){
		a[i] = new double[6];
	}

	MatrixSequence A(a,2,2,3);

	StaticMatrix AA = A(0);
	AA(0,0) = 1; AA(0,1) = 3; AA(0,2) = 5;
	AA(1,0) = 2; AA(1,1) = 4; AA(1,2) = 6;

	double b[6] = {1,2,3,4,5,6};
	
	for(int i = 0;i<AA.rows();i++){
		for(int j = 0;j < AA.cols();j++){
			ASSERT_EQ(A(0,i,j),b[2*j+i]);
		}
	}
}

GTEST_TEST(MatrixSequenceTest, IndexTest3) {
	double a[2][6] = {{1,2,3,4,5,6}, {3,4,5,6,7,8}};
	double *a1[2];
	a1[0] = &a[0][0];
	a1[1] = &a[1][0];

	StaticMatrix B[] = {StaticMatrix(a1[0],2,3),StaticMatrix(a1[1],2,3)};
	MatrixSequence A(B,2);
	
	StaticMatrix A0 = A(0);
	StaticMatrix A1 = A(1);

	for(int i = 0;i<A0.size();i++){
		ASSERT_EQ(A0.data[i],a[0][i]);
	}

	for(int i = 0;i<A1.size();i++){
		ASSERT_EQ(A1.data[i],a[1][i]);
	}
}

}  // namespace test
}  // namespace dominic
}  // namespace solvers
}  // namespace drake
