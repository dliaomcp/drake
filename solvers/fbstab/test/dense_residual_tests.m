% @Author: Dominic Liao-McPherson
% @Date:   2019-01-24 14:38:19
% @Last Modified by:   Dominic Liao-McPherson
% @Last Modified time: 2019-01-24 19:20:58

clear all
close all
clc

alpha = 0.95;
sigma = 0.5;

H = [3,1;1,1];
f = [1;6];
A = [-1,0;0,1];
b = [0;-1];

[zopt,~,~,~,lambda] = quadprog(H,f,A,b);


x.z = [1;5];
x.v = [0.4;2];
x.y = b - A*x.z;

xbar.z = [-5;6];
xbar.v = [-9;1];
xbar.y = b - A*xbar.z;


%% FB residual 
r1.z = H*x.z + f + A'*x.v + sigma*(x.z-xbar.z);
r1.v = pfb(x.y + sigma*(x.v - xbar.v),x.v,alpha);

%% Natural residual

r2.z = H*x.z + f + A'*x.v;
r2.v = min(x.y,x.v);


%% Penalized natural residual
r3.z = H*x.z + f + A'*x.v;
r3.v = alpha*min(x.y,x.v) + (1-alpha)*max(0,x.y).*max(0,x.v);

% norm check
r3.norm = norm(r3.z)+ norm(r3.v);

%% Linear solver checks

% compute and factor the K matrix as a check
ys = x.y + sigma*(x.v - xbar.v);
[gamma,mu] = dphi(ys,x.v,alpha);
mus = sigma*gamma + mu;

K = H + sigma*eye(size(H)) + A'*diag(gamma./mus)*A;

L = chol(K,'lower');


%% apply the factorization to the penalized natural residual

r1 = -r3.z;
r2 = -r3.v;

rz = r1 - A'*(r2./mus);

dx.z = K\rz;
dx.v = (r2 + gamma.*(A*dx.z))./mus;

dx.y = b - A*dx.z;



% compute an element of the C-differential 
function [gamma,mu] = dphi(a,b,alpha)
	q = length(a);
	ztol = 1e-13;
	% computes an element from the C differential
	r = sqrt(a.^2 + b.^2);
	S0 =  r <= ztol; % zero elements
	S1 = a > 0 & b> 0; % positive orthant
	S2 = ~(S1 | S0); % the rest

	gamma = zeros(q,1);
	mu = zeros(q,1);
	for i = 1:q
		if S0(i)
			d = 1/sqrt(2);
			gamma(i) = alpha*(1- d);
			mu(i) = alpha*(1- d);
		elseif S1(i)
			gamma(i) = alpha*(1- a(i)/r(i)) + (1-alpha)*b(i);
			mu(i) = alpha*(1- b(i)/r(i)) + (1-alpha)*a(i);
		else % S2
			gamma(i) = alpha*(1- a(i)/r(i));
			mu(i) = alpha*(1- b(i)/r(i));

		end
	end
end

function y = pfb(a,b,alpha)
		yfb = a + b - sqrt(a.^2 + b.^2);
		ypen = max(0,a).*max(0,b);

		y = alpha* yfb + (1-alpha)*ypen;
end
