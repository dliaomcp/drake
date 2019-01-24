#include "drake/solvers/dominic/linalg/static_matrix.h"


int main(void){

	double a[4] = {1,2,3,4};

	drake::solvers::dominic::StaticMatrix A(a,2,2);

	std::cout << A;

	return 0;
}