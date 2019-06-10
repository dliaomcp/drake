#include "drake/solvers/fbstab/linalg/static_matrix.h"

#include <gtest/gtest.h>


#define DOUBLE_ABS_TOL 1e-14
namespace drake {
namespace solvers {
namespace fbstab {
namespace test {


GTEST_TEST(StaticMatrixTest, IndexTest) {
	double a[] = {2,10,10,5,9,2};
	StaticMatrix A(a,3,2);

	A(0,1) = 4;
	A(2,0) = 8;

	double B[] = {2,10,8,4,9,2};
	for(int i = 0;i<A.size();i++){
		ASSERT_EQ(A.data[i],B[i]);
	}
}

GTEST_TEST(StaticMatrixTest, SlicingTest){
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

GTEST_TEST(StaticMatrixTest, ReshapeTest){
	double a[] = {2,10,10,5,9,2};
	StaticMatrix A(a,3,2);
	double b[] = {2,10,10,5,9,2};
	StaticMatrix B(b,2,3);

	A.reshape(2,3);

	for(int i = 0;i<A.rows();i++){
		for(int j = 0;j<A.cols();j++){
			ASSERT_EQ(A(i,j),B(i,j));
		}
	}
}

GTEST_TEST(StaticMatrixTest, axpyTest){
	double a[] = {4,3,6};
	double b[] = {1,2,3};
	double c = -0.5;

	StaticMatrix y(a,3);
	StaticMatrix x(b,3);

	y.axpy(x,c);
	double result[] = {3.5,2.0,4.5};

	for(int i = 0;i<y.rows();i++){
		EXPECT_DOUBLE_EQ(y(i),result[i]);
	}
}

// test gemv
GTEST_TEST(StaticMatrixTest, gemvTest){

	double a1[] = {1,3,4,-5,5,2};
	double xx[] = {-5,6,2};
	double yy[] = {3,4};
	double a = 0.5;
	double b = 0.3;

	StaticMatrix x(xx,3,1);
	StaticMatrix y(yy,2,1);
	StaticMatrix A(a1,2,3);

	y.gemv(A,x,a,b);
	double yresult[] = {15.4,-19.3};

	for(int i = 0;i<y.rows();i++){
		EXPECT_DOUBLE_EQ(y(i),yresult[i]);
	}
	// with A transposed
	x.gemv(A,y,a,b,true);
	double xresult[] = {-22.75,80.85,19.8};

	for(int i = 0;i<x.rows();i++){
		EXPECT_DOUBLE_EQ(x(i),xresult[i]);
	}
}

// // test gemm
GTEST_TEST(StaticMatrixTest, gemmTest){
	double a1[] = {1,-5,-3,6,3,5,6,7,8};
	double b1[] = {3,5,-5,6,3,4};
	double c1[] = {3,5,-5,6,3,4};
	StaticMatrix A(a1,3,3);
	StaticMatrix B(b1,3,2);
	StaticMatrix C(c1,3,2);


	double a = -0.5;
	double b = 0.4;

	// C = a*A*B + b*C
	C.gemm(A,B,a,b);

	double res[] = {-0.3,19.5,10.0,-21.6,-2.3,-12.9};
	for(int i = 0;i<C.size();i++){
		EXPECT_NEAR(C.data[i],res[i],DOUBLE_ABS_TOL);
	}

	// C = a*A'*B + b*C
	C.gemm(A,B,a,b,true);
	double res2[] = {3.38,3.8,-2.5,1.86,-33.42,-49.66};
	for(int i = 0;i<C.size();i++){
		EXPECT_NEAR(C.data[i],res2[i],DOUBLE_ABS_TOL);
	}

	double a2[] = {2,4,5};
	double b2[] = {2,5};
	A.map(a2,3,1);
	B.map(b2,2,1);

	// C = a*A*B' + b*C
	C.gemm(A,B,a,b,false,true);
	double res3[] = {-0.6480,-2.48,-6,-4.256,-23.3680,-32.3640};
	for(int i = 0;i<C.size();i++){
		EXPECT_NEAR(C.data[i],res3[i],DOUBLE_ABS_TOL);
	}

	// C = a*A'*B' + b*C
	A.map(a1,2,3);
	B.map(b1,2,2);

	C.fill(1.0);

	C.gemm(A,B,a,b,true,true);
	double res4[] = {-13.6,19.9,8.4,12.9,-10.1,-22.1};
	for(int i = 0;i<C.size();i++){
		EXPECT_NEAR(C.data[i],res4[i],DOUBLE_ABS_TOL);
	}
}

// // test gram
GTEST_TEST(StaticMatrixTest, gramTest){
	
	// A'*diag(B)*A + C

	double a[] = {1,3,4,5,6,7,3,-5,3,-9};
	StaticMatrix A(a,5,2);

	double c[] = {4,5,3,2};
	StaticMatrix C(c,2,2);

	double b[] = {-5,3,5,8,2};
	StaticMatrix B(b,5);

	C.gram(A,B);

	double res[] = {378,-91,-93,143};
	for(int i = 0;i<C.size();i++){
		EXPECT_NEAR(C.data[i],res[i],DOUBLE_ABS_TOL);
	}
}

// // test cholesky
GTEST_TEST(StaticMatrixTest, CholTest){
	double a[] = {14,9,8,9,16,1,8,1,17};
	StaticMatrix A(a,3,3);

	A.llt();
	A.tril();

	double res[] = {3.74165738677394,2.40535117721182,2.13808993529940,
		0,3.19597961731387,-1.29627145317626,0,0,3.27845264541853};

	for(int i =0;i<A.size();i++){
		EXPECT_NEAR(A.data[i],res[i],DOUBLE_ABS_TOL);
	}

	// backsolves
	double b[12] = {0};
	StaticMatrix B(b,3,4);
	B.fill(2);

	// B = inv(L)*B
	B.LeftCholApply(A);

	double res2[] = {0.534522483824849,	0.223495078133837,	0.349815376609394,	0.534522483824849,	0.223495078133837,	0.349815376609394,	0.534522483824849,	0.223495078133837,	0.349815376609394,	0.534522483824849,	0.223495078133837,	0.349815376609394};

	for(int i =0;i<A.size();i++){
		EXPECT_NEAR(B.data[i],res2[i],DOUBLE_ABS_TOL);
	}

	// B = inv(L)'*B
	B.LeftCholApply(A,true);

	double res3[] = {0.00910865322055958,	0.113207547169811,	0.106701366297983,	0.00910865322055958,	0.113207547169811,	0.106701366297983,	0.00910865322055958,	0.113207547169811,	0.106701366297983,	0.00910865322055958,	0.113207547169811,	0.106701366297983};

	for(int i =0;i<A.size();i++){
		EXPECT_NEAR(B.data[i],res3[i],DOUBLE_ABS_TOL);
	}


	B.fill(2);
	B.reshape(4,3);

	B.RightCholApply(A);
	double res4[] = {-0.375427557017496,-0.375427557017496,-0.375427557017496,	-0.375427557017496,	0.873216607108218,	0.873216607108218,	0.873216607108218,	0.873216607108218,	0.610043888477358,	0.610043888477358,	0.610043888477358,	0.610043888477358};

	for(int i =0;i<A.size();i++){
		EXPECT_NEAR(B.data[i],res4[i],DOUBLE_ABS_TOL);
	}

	B.RightCholApply(A,true);
	double res5[] = {-0.100337235136643,	-0.100337235136643,	-0.100337235136643,	-0.100337235136643,	0.348739049437080,	0.348739049437080,	0.348739049437080,	0.348739049437080,	0.389401505382707,	0.389401505382707,	0.389401505382707,	0.389401505382707};

	for(int i =0;i<A.size();i++){
		EXPECT_NEAR(B.data[i],res5[i],DOUBLE_ABS_TOL);
	}
}

}  // namespace test
}  // namespace dominic
}  // namespace solvers
}  // namespace drake
