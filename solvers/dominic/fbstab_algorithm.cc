#include "drake/solvers/dominic/fbstab_algorithm.h"

#include <cmath>
#include <cstdio>

#include "drake/solvers/dominic/linalg/static_matrix.h"
#include "drake/solvers/dominic/dense_components.h"


namespace drake {
namespace solvers {
namespace fbstab {


	FBstabAlgorithm::FBstabAlgorithm(DenseVariable *x1,DenseVariable *x2, DenseVariable *x3, DenseVariable *x4, DenseResidual *r1, DenseResidual *r2, DenseLinearSolver *lin_sol, DenseFeasibilityCheck *fcheck){

		this->xk = x1;
		this->xi = x2;
		this->xp = x3;
		this->dx = x4;

		this->rk = r1;
		this->ri = r2;
		this->linear_solver = lin_sol;
		this->feas = fcheck;
	}

	void FBstabAlgorithm::DeleteComponents(){
		delete xk;
		delete xi;
		delete xp;
		delete dx;

		delete rk;
		delete ri;

		delete linear_solver;
		delete feas;
	}


	SolverOut FBstabAlgorithm::Solve(DenseData *qp_data,DenseVariable *x0){
		// harmonize the alpha value
		rk->alpha = alpha;
		ri->alpha = alpha;
		linear_solver->alpha = alpha;

		// Output structure
		struct SolverOut output = {
			MAXITERATIONS, // eflag
			0.0, // residual
			0, // prox iters
		    0 // newton iters
		};

		// TODO figure out how to pass in data properly
		// link the data object
		xk->LinkData(qp_data);
		xi->LinkData(qp_data);
		dx->LinkData(qp_data);
		xp->LinkData(qp_data);
		x0->LinkData(qp_data);

		rk->LinkData(qp_data);
		ri->LinkData(qp_data);

		linear_solver->LinkData(qp_data);

		// initialization
		x0->InitConstraintMargin();
		xk->Copy(*x0);
		xi->Copy(*x0);
		dx->Fill(1.0);

		// regularization parameter is fixed for now
		double sigma = sigma0;

		// initialize residuals
		rk->NaturalResidual(*xk);
		ri->Fill(0.0);
		double E0 = rk->Norm();
		double Ek = E0;
		// tolerances
		double inner_tol = FBstabAlgorithm::min(E0,inner_tol_max);
		inner_tol = FBstabAlgorithm::max(inner_tol,inner_tol_min);
		// counters
		int newton_iters = 0;
		int prox_iters = 0;

		if(display_level == ITER){
			IterHeader();
		}

		// prox loop *************************************
		for(int k = 0;k<max_prox_iters;k++){

			rk->NaturalResidual(*xk);
			Ek = rk->Norm();

			// update tolerance
			inner_tol = FBstabAlgorithm::min(inner_tol_multiplier*inner_tol,Ek);
			inner_tol = FBstabAlgorithm::max(inner_tol,inner_tol_min);

			// Solver stops if:
			// a) the desired precision is obtained
			// b) the iterations stall, ie., ||x(k) - x(k-1)|| \leq tol
			if(Ek <= abs_tol + E0*rel_tol || dx->Norm() <= stall_tol){
				output.eflag = SUCCESS;
				output.residual = Ek;
				output.newton_iters = newton_iters;
				output.prox_iters = prox_iters;
				x0->Copy(*xk);

				if(display_level >= FINAL){
					PrintFinal(prox_iters,newton_iters,output.eflag,*rk);
				}
				return output;
			}

			if(display_level == ITER_DETAILED){
				DetailedHeader(prox_iters,newton_iters,*rk);
			} else if(display_level == ITER){
				IterLine(prox_iters,newton_iters,*rk,*ri,inner_tol);
			}

			// TODO: add a divergence check
			xi->Copy(*xk);
			rk->PenalizedNaturalResidual(*xk);
			double Ekpen = rk->Norm();
			double t = 1.0; // linesearch parameter
			ClearBuffer(merit_values,kNonmonotoneLinesearch);
			// inner loop *************************************
			for(int i = 0;i<max_inner_iters;i++){
				// compute residuals
				ri->FBresidual(*xi,*xk,sigma);
				double Ei = ri->Norm();

				// std::cout << "Rz: " << ri->rz << "\n";
				// std::cout << "Rz: " << ri->rv << "\n";
				rk->PenalizedNaturalResidual(*xi);
				double Eo = rk->Norm();

				// The inner loop stops if:
				// a) The subproblem is solved to the prescribed 
				// tolerance and the outer residual is reduced
				// b) The outer residual cannot be decreased 
				// (happens if problem is infeasible)
				if((Ei <= inner_tol && Eo < Ekpen) || (Ei <= inner_tol_min)){
					if(display_level == ITER_DETAILED){
						DetailedFooter(inner_tol,*ri);
					}
					break;
				}

				if(display_level == ITER_DETAILED){
					DetailedLine(i,t,*ri);
				}

				if(newton_iters >= max_newton_iters){
					output.eflag = MAXITERATIONS;
					if(Eo < Ekpen){
						x0->Copy(*xi);
						output.residual = Eo;
					} else{
						x0->Copy(*xk);
						output.residual = Ekpen;
						rk->PenalizedNaturalResidual(*xk);
					}
					output.newton_iters = newton_iters;
					output.prox_iters = prox_iters;
					if(display_level >= FINAL){
						PrintFinal(prox_iters,newton_iters,output.eflag,*rk);
					}
					return output;
				}

				// evaluate and factor the iteration matrix K at xi
				linear_solver->Factor(*xi,*xk,sigma);

				// solve for the Newton step, i.e., dx = K\ri
				ri->Negate(); 
				linear_solver->Solve(*ri,dx);
				newton_iters++;
				// TODO: solver error handling

				// linesearch *************************************
				double mcurr = ri->Merit();
				ShiftAndInsert(merit_values, mcurr, 
					kNonmonotoneLinesearch);

				double m0 = VectorMax(merit_values,
					kNonmonotoneLinesearch);

				t = 1.0;
				for(int j = 0;j<max_linesearch_iters;j++){
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
						t *= beta;
				} // linesearch *************************************
				xi->axpy(*dx,t); // xi <- xi + t*dx
				// std::cout << dx->z << "\n";

			} // inner loop *************************************

			// make duals non-negative
			xi->ProjectDuals();

			// compute dx <- x(k+1) - x(k)
			dx->Copy(*xi);
			dx->axpy(*xk,-1.0);

			// check for infeasibility
			if(check_infeasibility){
				InfeasibilityStatus status = CheckInfeasibility(*dx);
				// std::cout << status << "\n";
				if(status != FEASIBLE){
					if(status == PRIMAL){
						output.eflag = PRIMAL_INFEASIBLE;
					} else if (status == DUAL){
						output.eflag = UNBOUNDED_BELOW;
					}
					output.residual = Ek;
					output.newton_iters = newton_iters;
					output.prox_iters = prox_iters;
					// set x0 = x(k+1) - x(k)
					dx->Copy(*xi);
					dx->axpy(*xk,-1.0);
					x0->Copy(*dx);
					if(display_level >= FINAL){
						PrintFinal(prox_iters,newton_iters,output.eflag,*rk);
					}

					return output;
				}
			}

			// x(k+1) = x(i)
			xk->Copy(*xi);

			prox_iters++;
		} // prox loop

		// timeout exit
		output.eflag = MAXITERATIONS;
		output.residual = Ek;
		output.newton_iters = newton_iters;
		output.prox_iters = prox_iters;
		x0->Copy(*xk);
		if(display_level >= FINAL){
			PrintFinal(prox_iters,newton_iters,output.eflag,*rk);
		}

		return output;
	}

	FBstabAlgorithm::InfeasibilityStatus FBstabAlgorithm::CheckInfeasibility(const DenseVariable &x){

		// call the variable classes checker
		feas->CheckFeasibility(x,infeas_tol);

		InfeasibilityStatus s = FEASIBLE;

		if(!feas->Primal()){
			s = PRIMAL;
		} if(!feas->Dual()){
			s = DUAL;
		} if(!feas->Dual() && !feas->Primal()){
			s = BOTH;
		}

		return s;
	}

	void FBstabAlgorithm::UpdateOption(const char* option, double value){
		if(strcmp(option,"abs_tol")){
			abs_tol = max(value,1e-14);
		} else if(strcmp(option,"rel_tol")){
			rel_tol = max(value,0.0);
		} else if(strcmp(option,"stall_tol")){
			stall_tol = max(value,1e-14);
		} else if(strcmp(option,"infeas_tol")){
			infeas_tol = max(value,1e-14);
		} else if(strcmp(option,"sigma0")){
			sigma0 = max(value,1e-14);
		} else if(strcmp(option,"alpha")){
			alpha = max(value,0.001);
			alpha = min(alpha,0.999);
		} else if(strcmp(option,"beta")){
			beta = max(value,0.1);
			beta = min(beta,0.99);
		} else if(strcmp(option,"eta")){
			eta = max(value,1e-12);
			eta = min(eta,0.499);
		} else if(strcmp(option,"inner_tol_multiplier")){
			inner_tol_multiplier = max(value,0.0001);
			inner_tol_multiplier = min(inner_tol_multiplier,0.99);
		} else if(strcmp(option,"inner_tol_max")){
			inner_tol_max = max(value,1e-8);
			inner_tol_max = min(inner_tol_max,100.0);
		} else if(strcmp(option,"inner_tol_min")){
			inner_tol_min = max(value,1e-14);
			inner_tol_min = min(inner_tol_min,1e-2);
		} else{
			printf("%s is not an option, no action taken\n",option);
		}
	}

	void FBstabAlgorithm::UpdateOption(const char* option, int value){
		if(strcmp(option,"max_newton_iters")){
			max_newton_iters = max(value,1);
		} else if(strcmp(option,"max_prox_iters")){
			max_prox_iters = max(value,1);
		} else if(strcmp(option,"max_inner_iters")){
			max_inner_iters = max(value,1);
		} else if(strcmp(option,"max_linesearch_iters")){
			max_linesearch_iters = max(value,1);
		} else{
			printf("%s is not an option, no action taken\n",option);
		}
	}

	// static functions
	bool FBstabAlgorithm::strcmp(const char *x, const char *y){
		for(int i = 0;x[i] != '\0' || y[i] != '\0';i++){
			if(x[i] != y[i]){
				return false;
			}
		}
		return true;
	}
	
	void FBstabAlgorithm::ShiftAndInsert(double *buffer, double x, int buff_size){

		for(int i = 1;i<buff_size;i++){
			buffer[i] = buffer[i-1];
		}

		buffer[0] = x;
	}

	double FBstabAlgorithm::VectorMax(double *vec, int length){
		double current_max = vec[0];

		for(int i = 0;i<length;i++){
			if(vec[i] >= current_max){
				current_max = vec[i];
			}
		}

		return current_max;
	}

	void FBstabAlgorithm::ClearBuffer(double *buffer, int buff_size){
		for(int i = 0;i< buff_size;i++){
			buffer[i] = 0.0;
		}
	}

	double FBstabAlgorithm::max(double a,double b){
		return a>b ? a : b;
	}

	double FBstabAlgorithm::min(double a, double b){
		return a<b ? a : b;
	}

	int FBstabAlgorithm::max(int a,int b){
		return a>b ? a : b;
	}

	int FBstabAlgorithm::min(int a, int b){
		return a<b ? a : b;
	}

	// printing 
	void FBstabAlgorithm::PrintFinal(int prox_iters, int newton_iters, ExitFlag eflag, const DenseResidual &r){
		printf("Optimization completed\n Exit code:");
		switch(eflag){
			case SUCCESS:
				printf(" Success\n");
				break;
			case DIVERGENCE:
				printf(" Divergence\n");
				break;
			case MAXITERATIONS:
				printf(" Iteration limit exceeded\n");
				break;
			case PRIMAL_INFEASIBLE:
				printf(" Primal Infeasibility\n");
				break;
			case UNBOUNDED_BELOW:
				printf(" Unbounded Below\n");
				break;
			default:
				printf(" ???\n");
		}

		printf("Proximal iterations: %d out of %d\n", prox_iters, max_prox_iters);
		printf("Newton iterations: %d out of %d\n", newton_iters,max_newton_iters);
		printf("%10s  %10s  %10s\n","|rz|","|rv|","Tolerance");
		printf("%10.4e  %10.4e  %10.4e\n",r.z_norm,r.v_norm,abs_tol);
	}

	void FBstabAlgorithm::IterHeader(){
		printf("%12s  %12s  %12s  %12s  %12s  %12s\n","prox iter","newton iters","|rz|","|rv|","Inner res","Inner tol");
	}

	void FBstabAlgorithm::IterLine(int prox_iters, int newton_iters, const DenseResidual &r, const DenseResidual &r_inner,double itol){
		printf("%12d  %12d  %12.4e  %12.4e  %12.4e  %12.4e\n",prox_iters,newton_iters,r.z_norm,r.v_norm,r_inner.Norm(),itol);
	}

	void FBstabAlgorithm::DetailedHeader(int prox_iters, int newton_iters, const DenseResidual &r){
		double t = r.Norm();
		printf("Begin Prox Iter: %d, Total Newton Iters: %d, Residual: %6.4e\n",prox_iters,newton_iters,t);

		printf("%10s  %10s  %10s  %10s\n","Iter","Step Size","|rz|","|rv|");
	}

	void FBstabAlgorithm::DetailedLine(int iter, double step_length, const DenseResidual &r){
		printf("%10d  %10e  %10e  %10e\n",iter,step_length,r.z_norm,r.v_norm);
	}

	void FBstabAlgorithm::DetailedFooter(double tol, const DenseResidual &r){
		printf("Exiting inner loop. Inner residual: %6.4e, Inner tolerance: %6.4e\n",r.Norm(),tol);
	}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake