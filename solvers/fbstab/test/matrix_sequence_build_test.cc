#include "drake/solvers/dominic/linalg/matrix_sequence.h"

#include "drake/solvers/dominic/linalg/static_matrix.h"

using namespace drake::solvers::fbstab;
using namespace std;

int main(){


	// sequence of 2, 2 x 3 matrices
	double a[2][6] = {{1,2,3,4,5,6}, {3,4,5,6,7,8}};
	double *a1[2];
	a1[0] = &a[0][0];
	a1[1] = &a[1][0];

	// another sequence
	double** b = new double*[2];
	for(int i = 0;i<2;i++){
		b[i] = new double[6];
	}

	MatrixSequence A(a1,2,2,3);

	// grab the elements of the sequence
	StaticMatrix A0 = A(0);
	StaticMatrix A1 = A(1);

	cout << A0 << endl;
	cout << A1 << endl;

	// copy
	MatrixSequence B(b,2,2,3);
	B.copy(A);
	StaticMatrix B0 = B(0);
	StaticMatrix B1 = B(1);

	cout << B0 << endl;
	cout << B1 << endl;

	for(int i =0;i<2;i++){
		delete[] b[i];
	}
	delete[] b;


	return 0;
}