#pragma once


namespace drake {
namespace solvers {
namespace fbstab {

// a data to store input data
struct QPData {
	double *H;
	double *f;
	double *A;
	double *b;
};

// struct to hold workspace memory
// flesh out later -> will depend on how much memory allocation is needed
struct Workspace {
	double * wv1;
}

// the main object for the C++ API
class FBstabDense {

 public:
 	// dynamically initializes component classes 
 	FBstabDense(int n,int q);

 	// solve an instance of the QP
 	// remember that the x0 passed to the inner solver needs to have its y field initialized
 	Solve(const struct QPdata &qp);

 	// destructor
 	~FBstabDense();

 private:
 	bool memory_allocated = true;
 	int n;
 	int q;
}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
