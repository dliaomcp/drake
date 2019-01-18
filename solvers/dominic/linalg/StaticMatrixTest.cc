#include "StaticMatrix.h"
#include "TestingUtils.h"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace testutils;

// TODO: Overhaul for its google test based
int main(){

	// test assignment *************************************
	double *a1 = new double[9];
	double d1[] = {0.5,2,3};
	StaticMatrix A(a1,3,3);
	StaticMatrix d(d1,3,1);
	A.rand();
	A(2,2) = 100;
	A.tril();

	// test reshaping and slicing *************************************

	cout << "A:\n";
	cout << A << endl;
	StaticMatrix Acol = A.col(0);
	cout << "First column of A:\n" << Acol << endl;
	StaticMatrix Arow = A.row(0);
	cout << "First row of A:\n" << Arow << endl;
	StaticMatrix ARE = A.getreshape(1,9);
	cout << "reshape(A, [1,9]) is:\n" << ARE << endl;
	Arow(2) = 4;
	Acol(1) = 6;
	cout << "After A(0,2) = 4, A(1,0) = 6 and column mods\n" << A << endl;

	A.ColScale(d);
	cout << "After column scaling\n" << A << endl;
	A.RowScale(d);
	cout << "After row scaling\n" << A << endl;
	MatrixXd A1(3,3);
	CopyEig(A,A1);

	// axpy *************************************
	double *b1 = new double[3];
	StaticMatrix x(b1,3);
	x.rand();

	double *b2 = new double[3];
	StaticMatrix y(b2,3);
	y.rand();

	double a = -1.0;
	
	MatrixXd x1(3,1);
	CopyEig(x,x1);

	MatrixXd y1(3,1);
	CopyEig(y,y1);

	y1 = y1 + a*x1;
	y.axpy(x,a);
	cout << "Testing y <- a*x + y, Error: \n";
	cout << DiffNorm(y,y1) << endl << endl;
	

	// gemv *************************************
	double b = 1;
	// not transposed
	A.rand();
	CopyEig(A,A1);
	x.rand();
	CopyEig(x,x1);
	y.rand();
	CopyEig(y,y1);

	cout << "Testing y <- a*A*x + b*y, Error: \n";
	y1 = a*A1*x1 + b*y1;
	y.gemv(A,x,a,b);

	cout << DiffNorm(y,y1) << "\n\n";

	// transposed
	cout << "Testing y <- a*A'*x + b*y, Error: \n";
	double *a2 = new double[6];
	StaticMatrix B(a2,2,3);
	B.rand();
	MatrixXd B1(2,3);
	CopyEig(B,B1);

	double *b3 = new double[2];
	StaticMatrix z(b3,2);
	z.rand(); 
	y.rand();
	CopyEig(y,y1);

	MatrixXd z1(2,1);
	CopyEig(z,z1);

	y1 = a*(B1.transpose())*z1 + b*y1;
	y.gemv(B,z,a,0.0,true);
	cout << DiffNorm(y,y1) << "\n\n";

	// test dot and norms *************************************
	z1 = x1.transpose()*y1;
	cout << "Testing dot(x,y), error is: ";
	cout << abs(StaticMatrix::dot(x,y) - z1(0)) << "\n\n";

	cout << "Testing 2 norm, error is :";
	cout << B.norm() - B1.norm() << "\n\n";
	cout << "Testing 1 norm: ";
	cout << B.asum() << "\n\n";

	// free memory
	delete[] a1;
	delete[] a2;
	delete[] b1;
	delete[] b2;
	delete[] b3;
	
	// gemm *************************************

	a1 = new double[9];
	b1 = new double[6];
	double* c1 = new double[6];
	A.map(a1,3,3);
	B.map(b1,3,2);
	StaticMatrix C(c1,3,2);

	MatrixXd AA(3,3);
	MatrixXd BB(3,2);
	MatrixXd CC(3,2);

	A.rand();
	B.rand();
	C.rand();
	CopyEig(C,CC);
	CopyEig(A,AA);
	CopyEig(B,BB);

	cout << "Testing C = aAB+bC, error:";
	C.gemm(A,B,a,b);
	CC = a*AA*BB + b*CC;
	cout << DiffNorm(C,CC) << "\n\n";

	// transA
	cout << "Testing C = a A'B + bC, error:";
	C.gemm(A,B,a,b,true);
	CC = a*(AA.transpose())*BB + b*CC;
	cout << DiffNorm(C,CC) << "\n\n";
	

	// trans B
	delete[] a1;
	delete[] b1;
	a1 = new double[3];
	b1 = new double[2];
	A.map(a1,3,1);
	B.map(b1,2,1);
	A.rand();
	B.rand();
	AA.resize(3,1);
	BB.resize(2,1);
	CopyEig(C,CC);
	CopyEig(A,AA);
	CopyEig(B,BB);

	CC = a*AA*(BB.transpose()) + b*CC;
	C.gemm(A,B,a,b,false,true);

	cout << "Testing C = a AB' + bC, Error: " << DiffNorm(C,CC) << "\n\n";
	
	delete[] a1;
	delete[] b1;

	// trans A,B
	a1 = new double[6];
	b1 = new double[4];
	A.map(a1,2,3);
	B.map(b1,2,2);
	A.rand();
	B.rand();

	AA.resize(2,3);
	BB.resize(2,2);
	CopyEig(C,CC);
	CopyEig(A,AA);
	CopyEig(B,BB);
	CC = a*(AA.transpose())*(BB.transpose()) + b*CC;
	C.gemm(A,B,a,b,true,true);

	cout << "Testing C = a A'B' + bC, Error: " << DiffNorm(C,CC) << "\n\n";

	delete[] a1;
	delete[] b1;
	delete[] c1;

	// Symmetric products *************************************

	a1 = new double[10];
	b1 = new double[5];
	c1 = new double[4];

	A.map(a1,5,2);
	C.map(c1,2,2);
	A.rand();
	AA.resize(5,2);
	CC.resize(2,2);


	CopyEig(C,CC);
	CopyEig(A,AA);
	

	CC = (AA.transpose())*AA + CC;
	C.gram(A);
	cout << "Testing C = A'*A, Error: " << DiffNorm(C,CC) << "\n\n";


	B.map(b1,5,1);
	B.rand();
	BB.resize(5,5);
	BB.fill(0);
	for(int i = 0; i< 5;i++)
		BB(i,i) = B(i);

	CC = (AA.transpose())*BB*AA + CC;
	C.gram(A,B);
	cout << "Testing C = A'*diag(B)*A + C, Error: " << DiffNorm(C,CC) << "\n\n";

	delete[] a1;
	delete[] b1;

	// Cholesky factorization *************************************
	double lmem[] = {3.8966,2.1881,1.1965,2.1551,
    2.1881,2.9966,0.6827,1.8861,
    1.1965,    0.6827,    1.7590,    0.5348,
    2.1551,    1.8861,    0.5348,    3.0955};

    A.map(lmem,4,4);
    Map<MatrixXd> AL(lmem,4,4);
    LLT<MatrixXd> LLTA(AL);
    A.llt();
    MatrixXd L = LLTA.matrixLLT();

    cout << "Testing Cholesky Factorizaton, Error: ";
    cout << DiffNorm(A,L) << endl;
    for(int j = 0;j<4;j++)
    	for(int i = 0;i< j;i++)
    		L(i,j) = 0.0;

	// Back solves *************************************

    double b4[] = {0.6541,    0.6892,    0.7482,    0.4505,
    0.0838,    0.2290,    0.9133,    0.1524,
    0.8258,    0.5383,    0.9961,    0.0782};
    B.map(b4,4,3);
    MatrixXd BE(3,4);
    BE << 0.6541,    0.6892,    0.7482,    0.4505,
    0.0838,    0.2290,    0.9133,    0.1524,
    0.8258,    0.5383,    0.9961,    0.0782;
    BE.transposeInPlace();

    //*************************************
    cout << "Testing B <- inv(L)*B, A = LL' \n";
   
    BE = (L.inverse())*BE;
    B.LeftCholApply(A);
    cout << "Error: " << DiffNorm(B,BE) << endl << endl;

    //*************************************
    cout << "Testing B <- inv(L)'*B, A = LL' \n";

    B.LeftCholApply(A,true);
    BE = ((L.inverse()).transpose())*BE;

    cout << "Error : " << DiffNorm(B,BE) << endl << endl;

    // *************************************

    B.reshape(3,4);
    BE.resize(3,4);

    cout << "Testing B <- B*inv(L) \n";
    B.RightCholApply(A);
    BE = BE*(L.inverse());

    cout << "Error : " << DiffNorm(B,BE) << endl << endl;


    // *************************************
    cout << "Testing B <- B*inv(L)' \n";

    B.RightCholApply(A,true);
    BE = BE*((L.inverse()).transpose());

    cout << "Error : " << DiffNorm(B,BE) << endl << endl;





	return 0;
}



