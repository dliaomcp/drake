% @Author: Dominic Liao-McPherson
% @Date:   2019-01-24 14:38:19
% @Last Modified by:   Dominic Liao-McPherson
% @Last Modified time: 2019-01-24 14:59:37

clear all
close all
clc



H = [3,1;1,1];
f = [1;6];
A = [-1,0;0,1];
b = [0;-1];

zopt = quadprog(H,f,A,b)

%% margins
x.z = ones(2,1);
x.y = b - A*x.z;
x.v = ones(2,1);

y.z = -1*ones(2,1);
y.y = b-A*y.z;
y.v = zeros(2,1);

x.y
y.y

% compute an axpy 
a = 0.35;
y.z = y.z + a* x.z;
y.v = y.v + a* x.v;

y.y = y.y + a*(x.y - b);
y.z
y.v
y.y