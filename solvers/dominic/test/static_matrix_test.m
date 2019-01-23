
%% Computes the math output for the GTEST unit testing of the StaticMatrix class

%% indexing
A = [2,5
     10,9
     10,2]

A(1,2) = 4;
A(3,1) = 8;

A(:)

%% slicing
A = [2,5
     10,9
     10,2]

Arow = A(2,:)
Acol = A(:,2)


reshape(A,[2,3])


%% axpy
x = [1,2,3]';

y = [4,3,6]';

a = -0.5;

y = a*x + y



%% gemv
y = [3;4];
x = [-5;6;2];
A = [1,4,5;3,-5,2];

a = 1/2;
b = 0.3;

y = a*A*x + b*y


x = a*A'*y + b*x

% gemm
A = [1,-5,-3,6,3,5,6,7,8];
A = reshape(A,[3,3]);
B = [3,5,-5,6,3,4];
B = reshape(B,[3,2]);
a = -0.5;
b = 0.4;

C = B;

C = a*A*B + b*C
C(:)

C = a*A'*B + b*C
C(:)


A = [2;4;5];
B = [2;5];

C = a*A*B'+b*C
C(:)

A = [1,-5,-3,6,3,5];
B = [3,5,-5,6];

A = reshape(A,[2,3]);
B = reshape(B,[2,2]);
C = ones(3,2);

C = a*A'*B'+b*C
C(:)


A = reshape([1,3,4,5,6,7,3,-5,3,-9],[5,2]);
C = reshape([4,5,3,2],[2,2]);
B = diag([-5,3,5,8,2]);

C = A'*B*A+C
C(:)



A = reshape([4,9,8,9,6,1,8,1,7],[3,3]) + 10*eye(3);

L = chol(A,'lower')



B = 2*ones(3,4);

B = inv(L)*B;
B1 = B(:)';


B = inv(L)'*B;
B2 = B(:)';



B = 2*ones(4,3);

B = B*inv(L);
B3 = B(:)';


B = B*inv(L)';
B4 = B(:)';








