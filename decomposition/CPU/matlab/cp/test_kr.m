
clear all;
addpath('/home/luhan/decomposition/CPU/matlab/tensorlab');
	a = 10;
	r = 5;
	A = single(rand(a,r));
	B = single(rand(a,r));
	C = single(rand(a*a,r));


	t1=clock;
	U1 = gpuArray(A);
	U2 = gpuArray(B);
	U3 = gpuArray(C);
	U3 = kr(U1,U2);	
	
	C = gather(U3);

