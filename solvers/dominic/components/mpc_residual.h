#pragma once

#include "drake/solvers/dominic/linalg/static_matrix.h"
#include "drake/solvers/dominic/linalg/matrix_sequence.h"
#include "drake/solvers/dominic/components/mpc_data.h"
#include "drake/solvers/dominic/components/mpc_variable.h"

namespace drake {
namespace solvers {
namespace fbstab {

class MPCResidual{
 public:

 	MPCResidual(QPsizeMPC size);
 	~MPCResidual();
 	void LinkData(MPCData *data);
 	void SetAlpha(double alpha);

 	void Fill(double a);
 	void Copy(const MPCResidual &x);
 	void Negate();
 	double Norm() const;
 	double Merit() const;
 	double AbsSum() const;

 	void FBresidual(const MPCVariable &x, const MPCVariable &xbar, double sigma);
 	void NaturalResidual(const MPCVariable &x);
 	void PenalizedNaturalResidual(const MPCVariable &x);
 	
 	StaticMatrix z_;
 	StaticMatrix l_;
 	StaticMatrix v_;

 	double z_norm = 0.0;
 	double v_norm = 0.0;
 	double l_norm = 0.0;

 private:
 	int N_, nx_, nu_, nc_;
 	int nz_, nl_, nv_;

 	double alpha_ = 0.95;
 	MPCData *data_ = nullptr;

 	static double pfb(double a, double b, double alpha);
 	static double max(double a, double b);
 	static double min(double a, double b);

 	bool memory_allocated_ = false;
};


}  // namespace fbstab
}  // namespace solvers
}  // namespace drake