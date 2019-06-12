#pragma once

#include "drake/solvers/fbstab/linalg/static_matrix.h"
#include "drake/solvers/fbstab/components/dense_data.h"


namespace drake {
namespace solvers {
namespace fbstab {

// stores primal-dual variables
// x = (z,v,y)
// TODO: More documentation
class DenseVariable{
 public:
	DenseVariable(DenseQPsize size);
	~DenseVariable();

	// links in a DenseData object
	void LinkData(DenseData *data);
	// x <- a*ones
	void Fill(double a);
	// set the y field = b - Az
	void InitializeConstraintMargin();
	// y <- a*x + y
	void ScaledAccumulate(const DenseVariable &x, double a);
	// deep copy
	void Copy(const DenseVariable &x);
	// projects inequality duals onto the nonnegative orthant
	void ProjectDuals();
	double Norm() const;
	double InfNorm() const;

	StaticMatrix& z(){ return z_; };
	StaticMatrix& v(){ return v_; };
	StaticMatrix& y(){ return y_; }; 

	friend std::ostream &operator<<(std::ostream& output, const DenseVariable &x);

 private:
	int n_,q_; // sizes
	DenseData *data_ = nullptr;
	bool memory_allocated_ = false;

	StaticMatrix z_; // primal
	StaticMatrix v_; // dual
	StaticMatrix y_; // ieq margin
};


}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
