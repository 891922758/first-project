
clear all;
addpath('/home/luhan/decomposition/CPU/matlab/tensorlab/');
for a = 20:20:20
	size_tens = [a,a,a];
	size_core = [a*0.1,a*0.1,a*0.1];
	r = a*0.1;
	[U0,S0] = lmlra_rnd(size_tens,size_core);
	U0{1} = single(rand(a,a*0.1));
	U0{2} = single(rand(a,a*0.1));
	U0{3} = single(rand(a,a*0.1));
	S0 = single(rand(r,r,r));
	T = lmlragen(U0,S0);

	t0=clock;
	T1_gpu = gpuArray(tens2mat(T,1));
	T2_gpu = gpuArray(tens2mat(T,2));
	T3_gpu = gpuArray(tens2mat(T,3));
	t1_gpu = T1_gpu*T1_gpu';
	t2_gpu = T2_gpu*T2_gpu';
	t3_gpu = T3_gpu*T3_gpu';

	[U1_gpu,~] = eig(t1_gpu);
	U1_gpu = U1_gpu(1:a,a-a*0.1+1:a);
	[U2_gpu,~] = eig(t2_gpu);
	U2_gpu = U2_gpu(1:a,a-a*0.1+1:a);
	[U3_gpu,~] = eig(t3_gpu);
	U3_gpu = U3_gpu(1:a,a-a*0.1+1:a);
	U{1} = gather(U1_gpu);
	U{2} = gather(U2_gpu);
	U{3} = gather(U3_gpu);

	S = tmprod(T,U,1:3,'T');
	display(S);
	t1=clock;
	res = lmlragen(U,S);
	error = norm(res(:)-T(:))/norm(T(:));
	fid = fopen('hosvd_tensorlab.txt','a');
	fprintf(fid,'%d  %g  %fs\n', a, error , etime(t1,t0));
	fclose(fid);
	clear all;
end

