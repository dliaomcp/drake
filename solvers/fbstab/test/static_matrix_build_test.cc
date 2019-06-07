#include "drake/solvers/dominic/linalg/static_matrix.h"

using namespace drake::solvers::fbstab;
using namespace std;

int main(void){

	double a[4] = {1,2,3,4};

	StaticMatrix A(a,2,2);

	std::cout << A;

	A.reshape(4,1);

	StaticMatrix B = A.subvec(1,2);

	cout << B;



	return 0;
}