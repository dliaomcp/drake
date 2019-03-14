#pragma once


namespace drake {
namespace solvers {
namespace fbstab {
namespace test {

// repeats a matrix N times, i.e., 
// Q0,Q1,Q2,... QN-1
// remember to free the memory
double** repmat(double *A, int m, int n, int N){

	double **a = new double*[N];
	for(int i = 0;i<N;i++){
		a[i] = new double[m*n];
		for(int j = 0;j<m*n;j++){
			a[i][j] = A[j];
		}
	}
	return a;
}

// frees the memory allocated with repmat
void free_repmat(double **x, int N){
	for(int i=0;i<N;i++){
		delete[] x[i];
	}
	delete[] x;
}

}  // namespace test
}  // namespace fbstab
}  // namespace solvers
}  // namespace drake