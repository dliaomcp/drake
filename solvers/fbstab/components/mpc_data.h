#pragma once

#include "drake/solvers/fbstab/linalg/static_matrix.h"
#include "drake/solvers/fbstab/linalg/matrix_sequence.h"

namespace drake {
namespace solvers {
namespace fbstab {


/**
 * Stores the size of the MPC QP, fields:
 * 
 * N:  horizon length
 * nx: number of states
 * nu: number of control input
 * nc: number of constraints per stage
 */
struct QPsizeMPC {
	int N;
	int nx;
	int nu;
	int nc;
};

/**
 * Represents data for quadratic programming problems of the following type (1):
 *
 * min.  \sum_{i=0}^N 1/2 [x(i)]' * [Q(i) S(i)'] [x(i)] + [q(i)]'*[x(i)]
 *                        [u(i)]    [S(i) R(i) ] [u(i)]   [r(i)]  [u(i)]
 * s.t.  x(i+1) = A(i)*x(i) + B(i) u(i) + c(i), i = 0 ... N-1
 *       x(0) = x0
 *       E(i)*x(i) + L(i)*u(i) + d(i) <= 0,     i = 0 ... N
 *
 * This is a specialization of the general form (2):
 *
 * min.  1/2 z'*H*z + f'*z
 * 
 * s.t.  Gz =  h
 *       Az <= b
 *
 * Contains storage and methods for implicitly working with the compact 
 * representation (2).
 * 
 */
class MPCData{
 public:

 	/**
	 * Save the problem data
	 *
	 * Most inputs are series of matrices stored using nested pointers, 
	 * i.e., Q[i] should be a pointer to an array that stores Qi in 
	 * column major format for i = 0, ..., N. (or sometimes N-1)
	 *
	 * @param[in] Q = Q0,Q1,..., QN, length(Q) = N+1, length(Q[i]) = nx*nx
	 * @param[in] R = R0,R1,..., RN, length(R) = N+1, length(R[i]) = nu*nu
	 * @param[in] S = S0,S1,..., SN, length(S) = N+1, length(S[i]) = nu*nx
	 * @param[in] q = q0,q1,..., qN, length(q) = N+1, length(q[i]) = nx
	 * @param[in] r = r0,r1,..., rN, length(r) = N+1, length(r[i]) = nu
	 *
	 * @param[in] A = A0,A1,..., AN-1, length(A) = N, length(A[i]) = nx*nx
	 * @param[in] B = B0,B1,..., BN-1, length(B) = N, length(B[i]) = nx*nu
	 * @param[in] c = c0,c1,..., cN-1, length(c) = N, length(c[i]) = nx
	 * @param[in] x0, length(x0) = nx
	 *
	 * @param[in] E = E0,E1,..., EN, length(E) = N+1, length(E[i]) = nc*nx
	 * @param[in] L = L0,L1,..., LN, length(L) = N+1, length(L[i]) = nc*nu
	 * @param[in] d = d0,d1,..., dN, length(d) = N+1, sizeof(d[i]) = nc
	 *
	 * @param[in] size QP size structure
	 * E.g., if Q0 =  [1 3],  Q1 = [5 7]
	 * 				  [2 4]        [6 8]
	 *
	 * Then Q[0] should point to {1,2,3,4} and Q[1] should point to {5,6,7,8}
	 *
	 */
 	MPCData(double **Q, double **R, double **S, double **q, 
 			double **r, double **A, double **B, double **c,
 			double **E, double **L, double **d, double *x0, 
 			QPsizeMPC size);

 	/**
 	 * Computes y <- a*H*x + b*y without forming H explicitly
 	 * @param[in] x Input vector, length(x) = (nx+nu)*(N+1)
 	 * @param[in] a Input scaling
 	 * @param[in] b Scaling
 	 * @param[both] y Output vector, length(y) = (nx+nu)*(N+1)
 	 */
 	void gemvH(const StaticMatrix &x, double a, double b, StaticMatrix *y);

 	/**
 	 * Computes y <- a*A*x + b*y without forming A explicitly
 	 * @param[in] x Input vector, length(x) = (nx+nu)*(N+1)
 	 * @param[in] a Input scaling
 	 * @param[in] b Scaling
 	 * @param[both] y Output vector, length(y) = nc*(N+1)
 	 */ 
 	void gemvA(const StaticMatrix &x, double a, double b, StaticMatrix *y); 

 	/**
 	 * Computes y <- a*G*x + b*y without forming G explicitly
 	 * @param[in] x Input vector, length(x) = (nx+nu)*(N+1)
 	 * @param[in] a Input scaling
 	 * @param[in] b Scaling
 	 * @param[both] y Output vector, length(y) = nx*(N+1)
 	 */ 
 	void gemvG(const StaticMatrix &x, double a, double b, StaticMatrix *y); 

 	/**
 	 * Computes y <- a*A'*x + b*y without forming A explicitly
 	 * @param[in] x Input vector, length(x) = nc*(N+1)
 	 * @param[in] a Input scaling
 	 * @param[in] b Scaling
 	 * @param[both] y Output vector, length(y) = (nx+nu)*(N+1)
 	 */ 
 	void gemvAT(const StaticMatrix &x, double a, double b, StaticMatrix *y); 

 	/**
 	 * Computes y <- a*G'*x + b*y without forming G explicitly
 	 * @param[in] x Input vector, length(x) = (nx)*(N+1)
 	 * @param[in] a Input scaling
 	 * @param[in] b Scaling
 	 * @param[both] y Output vector, length(y) = (nx+nu)*(N+1)
 	 */ 
	void gemvGT(const StaticMatrix &x, double a, double b, StaticMatrix *y);

	/**
	 * Computes y <- a*f + y
	 * @param[in] a Scaling factor
	 * @param[both] y Output vector, length(y) = (nx+nu)*(N+1)
	 */
	void axpyf(double a, StaticMatrix *y);

	/**
	 * Computes y <- a*h + y
	 * @param[in] a Scaling factor
	 * @param[both] y Output vector, length(y) = nx*(N+1)
	 */
	void axpyh(double a, StaticMatrix *y);

	/**
	 * Computes y <- a*b + y
	 * @param[in] a Scaling factor
	 * @param[both] y Output vector, length(y) = nc*(N+1)
	 */
	void axpyb(double a, StaticMatrix *y);

 private:
 	int N_;
 	int nx_;
 	int nu_;
 	int nc_;

 	int nz_;
 	int nl_;
 	int nv_;

 	// storage
 	MatrixSequence Q_,R_,S_,q_,r_;
 	MatrixSequence A_,B_,c_;
 	MatrixSequence E_,L_,d_;
 	StaticMatrix x0_;

 	friend class RicattiLinearSolver;
};

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake
