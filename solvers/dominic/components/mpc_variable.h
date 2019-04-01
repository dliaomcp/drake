#pragma once

#include "drake/solvers/dominic/linalg/static_matrix.h"
#include "drake/solvers/dominic/linalg/matrix_sequence.h"
#include "drake/solvers/dominic/components/mpc_data.h"

namespace drake {
namespace solvers {
namespace fbstab {

class MPCVariable{
 public:
 	MPCVariable(QPsizeMPC size);
 	MPCVariable(QPsizeMPC size, double *z, double *l, double *v, double *y);
 	~MPCVariable();

 	void LinkData(MPCData *data);
 	void Fill(double a);
 	void InitConstraintMargin();
 	void axpy(const MPCVariable &x, double a);
 	void Copy(const MPCVariable &x);
 	void ProjectDuals();
 	double Norm();
 	double InfNorm();

 	StaticMatrix z_; // decision variables [x0,u0,x1,u1 ... xN,uN]
 	StaticMatrix l_; // co-states [l0, ..., lN];
 	StaticMatrix v_; // ieq-duals [v0,... vN];
 	StaticMatrix y_; // ieq margin, [y0, ..., yN]

 	friend std::ostream &operator<<(std::ostream& output, const MPCVariable &x);

 private:
 	int N_, nx_, nu_, nc_;
 	int nz_, nl_, nv_;
 	bool memory_allocated_ = false;
 	MPCData *data_ = nullptr;
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake