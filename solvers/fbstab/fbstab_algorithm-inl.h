#pragma once

namespace drake {
namespace solvers {
namespace fbstab {

template <class Variable, class Residual, class Data, class LinearSolver, class Feasibility>
FBstabAlgorithm<Variable,Residual,Data,LinearSolver,Feasibility> ::FBstabAlgorithm(Variable *x1,Variable *x2, Variable *x3, Variable *x4, Residual *r1, Residual *r2, LinearSolver *lin_sol, Feasibility *fcheck){
	if(x1 == nullptr || x2 == nullptr || x3 == nullptr || x4 == nullptr){
		throw std::runtime_error("A Variable supplied to FBstabAlgorithm is invalid.");
	}
	if(r1 == nullptr || r2 == nullptr){
		throw std::runtime_error("A Residual supplied to FBstabAlgorithm is invalid");
	}
	if(lin_sol == nullptr){
		throw std::runtime_error("The LinearSolver supplied to FBstabAlgorithm is invalid.");
	}
	if(fcheck == nullptr){
		throw std::runtime_error("The Feasibility object supplied to FBstabAlgorithm is invalid.");
	}

	xk_ = x1;
	xi_ = x2;
	xp_ = x3;
	dx_ = x4;

	rk_ = r1;
	ri_ = r2;

	linear_solver_ = lin_sol;
	feasibility_ = fcheck;
}

// TODO(dliaomcp@umich.edu): Enable printing to a log file rather than just stdout
template <class Variable, class Residual, class Data, class LinearSolver, class Feasibility>
SolverOut FBstabAlgorithm<Variable,Residual,Data,LinearSolver,Feasibility>::Solve(const Data* qp_data, Variable* x0){
	// Make sure the linear solver and residuals objects are using the same value
	// for the alpha parameter.
	rk_->SetAlpha(alpha_);
	ri_->SetAlpha(alpha_);
	linear_solver_->SetAlpha(alpha_);

	struct SolverOut output = {
		MAXITERATIONS, // exit flag
		0.0, // residual
		0, // prox iters
	    0 // newton iters
	};

	// Supply a pointer to the data object
	xk_->LinkData(qp_data);
	xi_->LinkData(qp_data);
	dx_->LinkData(qp_data);
	xp_->LinkData(qp_data);
	x0->LinkData(qp_data);
	rk_->LinkData(qp_data);
	ri_->LinkData(qp_data);
	linear_solver_->LinkData(qp_data);
	feasibility_->LinkData(qp_data);

	// initialization
	double sigma_ = sigma0_;

	x0->InitializeConstraintMargin();
	xk_->Copy(*x0);
	xi_->Copy(*x0);
	dx_->Fill(1.0);

	rk_->NaturalResidual(*xk_);
	ri_->Fill(0.0);
	double E0 = rk_->Norm(); 
	double Ek = E0;

	double inner_tol = FBstabAlgorithm::min(E0,inner_tol_max_);
	inner_tol = FBstabAlgorithm::max(inner_tol,inner_tol_min_);

	newton_iters_ = 0;
	prox_iters_ = 0;

	PrintIterHeader();

	// main prox loop *************************************
	for(int k = 0; k < max_prox_iters_; k++){
		rk_->NaturalResidual(*xk_);
		Ek = rk_->Norm();

		// The solver stops if:
		// a) the desired precision is obtained
		// b) the iterations stall, ie., ||x(k) - x(k-1)|| \leq tol
		if(Ek <= abs_tol_ + E0*rel_tol_ || dx_->Norm() <= stall_tol_){
			output.eflag = SUCCESS;
			output.residual = Ek;
			output.newton_iters = newton_iters_;
			output.prox_iters = prox_iters_;
			x0->Copy(*xk_);

			PrintIterLine(prox_iters_,newton_iters_,*rk_,*ri_,inner_tol);
			PrintFinal(prox_iters_,newton_iters_,output.eflag,*rk_);

			return output;
		} else {
			PrintDetailedHeader(prox_iters_,newton_iters_,*rk_);
			PrintIterLine(prox_iters_,newton_iters_,*rk_,*ri_,inner_tol);
		}

		// TODO(dliaomcp@umich.edu): add a divergence check

		// Update subproblem tolerance
		inner_tol = FBstabAlgorithm::min(inner_tol_multiplier_*inner_tol,Ek);
		inner_tol = FBstabAlgorithm::max(inner_tol,inner_tol_min_);

		// Solve the proximal subproblem
		xi_->Copy(*xk_);
		rk_->PenalizedNaturalResidual(*xk_);
		double Ekpen = rk_->Norm();
		double Eo = SolveSubproblem(xi_,xk_,inner_tol,sigma_,Ekpen);

		if(newton_iters_ >= max_newton_iters_){ // if iteration limit exceeded
			output.eflag = MAXITERATIONS;
			if(Eo < Ekpen){
				x0->Copy(*xi_);
				output.residual = Eo;
			} else{
				x0->Copy(*xk_);
				output.residual = Ekpen;
				rk_->PenalizedNaturalResidual(*xk_);
			}
			output.newton_iters = newton_iters_;
			output.prox_iters = prox_iters_;
			PrintFinal(prox_iters_,newton_iters_,output.eflag,*rk_);
			return output;
		}

		// compute dx <- x(k+1) - x(k)
		dx_->Copy(*xi_);
		dx_->axpy(*xk_,-1.0);

		// check for infeasibility
		if(check_feasibility_){
			InfeasibilityStatus status = CheckInfeasibility(*dx_);
			if(status != FEASIBLE){
				if(status == PRIMAL){
					output.eflag = INFEASIBLE;
				} else if (status == DUAL){
					output.eflag = UNBOUNDED_BELOW;
				}
				output.residual = Ek;
				output.newton_iters = newton_iters_;
				output.prox_iters = prox_iters_;
				x0->Copy(*dx_);
				PrintFinal(prox_iters_,newton_iters_,output.eflag,*rk_);
				return output;
			}
		}

		// x(k+1) = x(i)
		xk_->Copy(*xi_);
		prox_iters_++;
	} // end proximal loop

	// Execute a timeout exit
	output.eflag = MAXITERATIONS;
	output.residual = Ek;
	output.newton_iters = newton_iters_;
	output.prox_iters = prox_iters_;
	x0->Copy(*xk_);

	PrintFinal(prox_iters_,newton_iters_,output.eflag,*rk_);
	return output;
}

template <class Variable, class Residual, class Data, class LinearSolver, class Feasibility>
double FBstabAlgorithm<Variable,Residual,Data,LinearSolver,Feasibility>::SolveSubproblem(Variable *x,Variable *xbar, 
	double tol, double sigma, double Eouter){

	double Eo = 0; // residual
	double t = 1.0; // linesearch parameter
	ClearBuffer(merit_values_,kNonmonotoneLinesearch);
	
	for(int i = 0; i < max_inner_iters_; i++){
		// compute inner residual
		ri_->InnerResidual(*x,*xbar,sigma);
		double Ei = ri_->Norm();

		// compute outer residual
		rk_->PenalizedNaturalResidual(*x);
		Eo = rk_->Norm();

		// The inner loop stops if:
		// a) The subproblem is solved to the prescribed 
		// tolerance and the outer residual is reduced
		// b) The outer residual cannot be decreased 
		// (happens if problem is infeasible)
		if((Ei <= tol && Eo < Eouter) || (Ei <= inner_tol_min_)){
			PrintDetailedLine(i,t,*ri_);
			PrintDetailedFooter(tol,*ri_);
			break;
		} else{
			PrintDetailedLine(i,t,*ri_);
		}
		if(newton_iters_ >= max_newton_iters_){
			break;
		}

		// solve for the Newton step
		linear_solver_->Factor(*x,*xbar,sigma);
		// TODO(dliaomcp@umich.edu): Add solver error handling
		ri_->Negate(); 
		linear_solver_->Solve(*ri_,dx_);
		newton_iters_++;
		
		// linesearch *************************************
		double current_merit = ri_->Merit();
		ShiftAndInsert(merit_values_, current_merit, kNonmonotoneLinesearch);
		double m0 = VectorMax(merit_values_, kNonmonotoneLinesearch);
		t = 1.0;

		for(int j = 0; j < max_linesearch_iters_; j++){
			// Compute trial point xp = x + t*dx
			// and evaluate the merit function at xp
			xp_->Copy(*x);
			xp_->axpy(*dx_,t);
			ri_->InnerResidual(*xp_,*xbar,sigma);
			double mp = ri_->Merit();

			// Armijo descent check
			if(mp <= m0 - 2.0*t*eta_*current_merit) {
				break;
			} else {
				t *= beta_;
			}
		} // end linesearch
		x->axpy(*dx_,t); // x <- x + t*dx
	} 
	// make duals non-negative
	x->ProjectDuals();

	return Eo;
}


template <class Variable, class Residual, class Data, class LinearSolver, class Feasibility>
typename FBstabAlgorithm<Variable,Residual,Data,LinearSolver,Feasibility>::InfeasibilityStatus FBstabAlgorithm<Variable,Residual,Data,LinearSolver,Feasibility>::CheckInfeasibility(const Variable &x){
	feasibility_->ComputeFeasibility(x,infeasibility_tol_);

	InfeasibilityStatus status = FEASIBLE;
	if(!feasibility_->IsPrimalFeasible()){
		status = PRIMAL;
	} if(!feasibility_->IsDualFeasible()){
		status = DUAL;
	} if(!feasibility_->IsDualFeasible() && !feasibility_->IsPrimalFeasible()){
		status = BOTH;
	}
	return status;
}

template <class Variable, class Residual, class Data, class LinearSolver, class Feasibility>
void FBstabAlgorithm<Variable,Residual,Data,LinearSolver,Feasibility>::UpdateOption(const char* option, double value){
	if(strcmp(option,"abs_tol")){
		abs_tol_ = max(value,1e-14);
	} else if(strcmp(option,"rel_tol")){
		rel_tol_ = max(value,0.0);
	} else if(strcmp(option,"stall_tol")){
		stall_tol_ = max(value,1e-14);
	} else if(strcmp(option,"infeas_tol")){
		infeasibility_tol_ = max(value,1e-14);
	} else if(strcmp(option,"sigma0")){
		sigma0_ = max(value,1e-14);
	} else if(strcmp(option,"alpha")){
		alpha_ = max(value,0.001);
		alpha_ = min(alpha_,0.999);
	} else if(strcmp(option,"beta")){
		beta_ = max(value,0.1);
		beta_ = min(beta_,0.99);
	} else if(strcmp(option,"eta")){
		eta_ = max(value,1e-12);
		eta_ = min(eta_,0.499);
	} else if(strcmp(option,"inner_tol_multiplier")){
		inner_tol_multiplier_ = max(value,0.0001);
		inner_tol_multiplier_ = min(inner_tol_multiplier_,0.99);
	} else if(strcmp(option,"inner_tol_max")){
		inner_tol_max_ = max(value,1e-8);
		inner_tol_max_ = min(inner_tol_max_,100.0);
	} else if(strcmp(option,"inner_tol_min")){
		inner_tol_min_ = max(value,1e-14);
		inner_tol_min_ = min(inner_tol_min_,1e-2);
	} else{
		printf("%s is not an option, no action taken\n",option);
	}
}

template <class Variable, class Residual, class Data, class LinearSolver, class Feasibility>
void FBstabAlgorithm<Variable,Residual,Data,LinearSolver,Feasibility>::UpdateOption(const char* option, int value){
	if(strcmp(option,"max_newton_iters")){
		max_newton_iters_ = max(value,1);
	} else if(strcmp(option,"max_prox_iters")){
		max_prox_iters_ = max(value,1);
	} else if(strcmp(option,"max_inner_iters")){
		max_inner_iters_ = max(value,1);
	} else if(strcmp(option,"max_linesearch_iters")){
		max_linesearch_iters_ = max(value,1);
	} else{
		printf("%s is not an option, no action taken\n",option);
	}
}

template <class Variable, class Residual, class Data, class LinearSolver, class Feasibility>
void FBstabAlgorithm<Variable,Residual,Data,LinearSolver,Feasibility>
::UpdateOption(const char* option, bool value){
	if(strcmp(option,"check_feasibility")){
		check_feasibility_ = value;
	} else{
		printf("%s is not an option, no action taken\n",option);
	}
}

// static functions *************************************
template <class Variable, class Residual, class Data, class LinearSolver, class Feasibility>
bool FBstabAlgorithm<Variable,Residual,Data,LinearSolver,Feasibility>::strcmp(const char *x, const char *y){
	for(int i = 0;x[i] != '\0' || y[i] != '\0';i++){
		if(x[i] != y[i]){
			return false;
		}
	}
	return true;
}

template <class Variable, class Residual, class Data, class LinearSolver, class Feasibility>
void FBstabAlgorithm<Variable,Residual,Data,LinearSolver,Feasibility>::ShiftAndInsert(double *buffer, double x, int buff_size){
	for(int i = 1;i<buff_size;i++){
		buffer[i] = buffer[i-1];
	}
	buffer[0] = x;
}

template <class Variable, class Residual, class Data, class LinearSolver, class Feasibility>
double FBstabAlgorithm<Variable,Residual,Data,LinearSolver,Feasibility>::VectorMax(double *vec, int length){
	double current_max = vec[0];
	for(int i = 0;i<length;i++){
		if(vec[i] >= current_max){
			current_max = vec[i];
		}
	}
	return current_max;
}

template <class Variable, class Residual, class Data, class LinearSolver, class Feasibility>
void FBstabAlgorithm<Variable,Residual,Data,LinearSolver,Feasibility>::ClearBuffer(double *buffer, int buff_size){
	for(int i = 0;i< buff_size;i++){
		buffer[i] = 0.0;
	}
}

// printing *************************************
template <class Variable, class Residual, class Data, class LinearSolver, class Feasibility>
void FBstabAlgorithm<Variable,Residual,Data,LinearSolver,Feasibility>
::PrintFinal(int prox_iters, int newton_iters, ExitFlag eflag, const Residual &r){

	if(display_level_ >= FINAL){
		printf("Optimization completed!  Exit code:");
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
			case INFEASIBLE:
				printf(" Primal Infeasibility\n");
				break;
			case UNBOUNDED_BELOW:
				printf(" Unbounded Below\n");
				break;
			default:
				printf(" ???\n");
		}
		printf("Proximal iterations: %d out of %d\n", prox_iters, max_prox_iters_);
		printf("Newton iterations: %d out of %d\n", newton_iters,max_newton_iters_);
		printf("%10s  %10s  %10s  %10s\n","|rz|","|rl|","|rv|","Tolerance");
		printf("%10.4e  %10.4e  %10.4e  %10.4e\n",r.z_norm(),r.l_norm(),r.v_norm(),abs_tol_);
		printf("\n");
	}
}

template <class Variable, class Residual, class Data, class LinearSolver, class Feasibility>
void FBstabAlgorithm<Variable,Residual,Data,LinearSolver,Feasibility>
::PrintIterHeader(){
	if(display_level_ == ITER){
		printf("%12s  %12s  %12s  %12s  %12s  %12s  %12s\n","prox iter","newton iters","|rz|","|rl|","|rv|","Inner res","Inner tol");
	}
}

template <class Variable, class Residual, class Data, class LinearSolver, class Feasibility>
void FBstabAlgorithm<Variable,Residual,Data,LinearSolver,Feasibility>
::PrintIterLine(int prox_iters, int newton_iters, const Residual &rk, const Residual &ri,double itol){
	if(display_level_ == ITER){
		printf("%12d  %12d  %12.4e  %12.4e  %12.4e  %12.4e  %12.4e\n",prox_iters,newton_iters,rk.z_norm(),rk.l_norm(),rk.v_norm(),ri.Norm(),itol);
	}
}

template <class Variable, class Residual, class Data, class LinearSolver, class Feasibility>
void FBstabAlgorithm<Variable,Residual,Data,LinearSolver,Feasibility>
::PrintDetailedHeader(int prox_iters, int newton_iters, const Residual &r){
	if(display_level_ == ITER_DETAILED){
		double t = r.Norm();
		printf("Begin Prox Iter: %d, Total Newton Iters: %d, Residual: %6.4e\n",prox_iters,newton_iters,t);
		printf("%10s  %10s  %10s  %10s  %10s\n","Iter","Step Size","|rz|","|rv|","|rl|");
	}
}

template <class Variable, class Residual, class Data, class LinearSolver, class Feasibility>
void FBstabAlgorithm<Variable,Residual,Data,LinearSolver,Feasibility>
::PrintDetailedLine(int iter, double step_length, const Residual &r){
	if(display_level_ == ITER_DETAILED){
		printf("%10d  %10e  %10e  %10e  %10e\n",iter,step_length,r.z_norm(),r.v_norm(),r.l_norm());
	}
}

template <class Variable, class Residual, class Data, class LinearSolver, class Feasibility>
void FBstabAlgorithm<Variable,Residual,Data,LinearSolver,Feasibility>
::PrintDetailedFooter(double tol, const Residual &r){
	if(display_level_ == ITER_DETAILED){
	printf("Exiting inner loop. Inner residual: %6.4e, Inner tolerance: %6.4e\n",r.Norm(),tol);
	}
}

}  // namespace fbstab
}  // namespace solvers
}  // namespace drake