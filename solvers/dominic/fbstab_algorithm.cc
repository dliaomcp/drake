#include "drake/solvers/dominic/fbstab_algorithm.h"

#include <cmath>

#include "drake/solvers/dominic/linalg/static_matrix.h"
#include "drake/solvers/dominic/dense_components"


namespace drake {
namespace solvers {
namespace fbstab {


	FBstabAlgorithm::FBstabAlgorithm(DenseVariable *x1,DenseVariable *x2,DenseVariable *x3, DenseVariable *x4, DenseResidual *r1, DenseResidual *r2, DenseLinearSolver *linear_solver){

		this->xk = x1;
		this->xi = x2;
		this->xp = x3;
		this->dx = x4;

		this->rk = r1;
		this->ri = r2;
		this->linear_solver = linear_solver;

	}

	SolverOut FBstabAlgorithm::Solve(const DenseData &qp_data,DenseVariable *x0){
		// harmonize the alpha value
		rk->alpha = alpha;
		ri->alpha = alpha;
		linear_solver->alpha = alpha;

		// Output structure
		struct SolverOut output = {
			MAXITERATIONS,
			0.0,
			0,
		    0
		};

		// link the data object
		xk->LinkData(&qp_data);
		xi->LinkData(&qp_data);
		dx->LinkData(&qp_data);
		xp->LinkData(&qp_data);
		x0->LinkData(&qp_data);

		rk->LinkData(&qp_data);
		ri->LinkData(&qp_data);

		linear_solver->LinkData(&qp_data);

		// compute the constraint margin for the initial guess
		x0->InitConstraintMargin();

		// initialization
		xk->Copy(*x0);
		xi->Copy(*x0);
		dx->Fill(1.0);

		// regularization parameter is fixed for now
		// this may change in the future
		double sigma = sigma0;

		// initialize outer loop residual
		rk->NaturalResidual(*xk);
		double E0 = rk->Norm();
		double Ek = rk->Norm();
		// tolerances
		double inner_tol = FBstabAlgorithm::min(E0,inner_tol_max);
		inner_tol = FBstabAlgorithm::max(inner_tol,inner_tol_min);

		int newton_iters = 0;
		int prox_iters = 0;
		// prox loop *************************************
		for(int k = 0;k<max_prox_iters;k++){

			// termination check
			// Solver stops if:
			// a) the desired precision is obtained
			// b) the iterations stall, ie., ||x(k) - x(k-1)|| \leq tol
			rk->NaturalResidual(*xk);
			Ek = rk->Norm();
			if(Ek <= abs_tol + E0*rel_tol || dx->Norm() <= stall_tol){
				output.eflag = SUCCESS;
				output.residual = Ek;
				output.newton_iters = newton_iters;
				output.prox_iters = prox_iters;

				x0->copy(*xk);
				
				return output;
			}

			// TODO: add a divergence check

			// used in inner loop termiantion check
			rk->PenalizedNaturalResidual(*xk);
			double Ekpen = rk->Norm();

			// pfb loop *************************************
			for(int i = 0;k<max_inner_iters;i++){

				// compute residuals
				ri->FBresidual(*xi,*xk,sigma);
				rk->PenalizedNaturalResidual(*xi);

				// termination check
				// Inner loop stops if:
				// a) Subproblem is solved to the perscribed 
				// tolerance and the outer residual is reduced
				// b) Outer residual cannot be decreased 
				// (happens if problem is infeasible)

				double Ei = ri->Norm();
				double Eo = rk->Norm();
				if( (Ei <= inner_tol && Eo < Ekpen) 
					|| (Ei <= inner_tol_min) ){
					// TODO: add print statements
					break;
				}

				// evaluate and factor the iteration matrix K at xi
				linear_solver->Factor(*xi,sigma);

				// solve for the Newton step, i.e., dx = K\ri
				ri->Negate(); 
				bool solve_flag = linear_solver->Solve(*ri,dx);
				newton_iters++;

				// TODO: solver error handling

				// linesearch *************************************
				double mcurr = ri->Merit();
				double m0 = VectorMax(merit_values,
					kNonmonotoneLinesearch);
				ShiftAndInsert(merit_values, mcurr, 
					kNonmonotoneLinesearch);

				double t = 1.0;
				for(int j = 0;k<max_linesearch_iters;j++){
					// xp = x + t*dx
					xp->Copy(*xi);
					xp->axpy(*dx,t);

					// evaluate the merit function at xp
					ri->FBresidual(*xp,*xk,sigma);
					double mp = ri->Merit();

					// acceptance check
					if(mp <= m0 - 2.0*t*eta*mcurr) 
						break;
					else 
						t = beta*t;
				} // end linesearch loop

				// xi <- xi + t*dx
				xi->axpy(*dx,t);
			} // pfb loop

			// make duals non-negative
			xi->ProjectDuals();

			// compute dx <- x(k+1) - x(k) = x(i) - x(k)
			dx->Copy(*xi);
			dx->axpy(xk,-1.0);

			// x(k+1) = x(i)
			xk->Copy(*xi);

			// check for infeasibility
			if(check_infeasibility){

			}

			// increment counter
			prox_iters++;
		} // prox loop


		// timeout exit
		output.eflag = MAXITERATIONS;
		output.residual = Ek;
		output.newton_iters = newton_iters;
		output.prox_iters = prox_iters;
		return output;
	}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake