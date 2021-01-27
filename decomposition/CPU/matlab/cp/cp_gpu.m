
clear all;
addpath('/home/luhan/decomposition/CPU/matlab/tensorlab');

for a = 100:100:1200
	size_tens = [a,a,a];
	r = a*0.1;
	U0 = cpd_rnd(size_tens, a*0.1);
	U0{1} = single(rand(a,r));
	U0{2} = single(rand(a,r));
	U0{3} = single(rand(a,r));
	T = cpdgen(U0);

	U = cpd_rnd(size_tens,a*0.1);
	U{1} = single(rand(a,r));
	U{2} = single(rand(a,r));
	U{3} = single(rand(a,r));

	t1=clock;
	U1_gpu = gpuArray(U{1});
	U2_gpu = gpuArray(U{2});
	U3_gpu = gpuArray(U{3});
	T1_gpu = gpuArray(tens2mat(T,1));
	T2_gpu = gpuArray(tens2mat(T,2));
	T3_gpu = gpuArray(tens2mat(T,3));

	for i =1:10
		U32_gpu = gpuArray(kr(U{3},U{2}));
		U1_gpu = T1_gpu*U32_gpu*pinv((U3_gpu'*U3_gpu).*(U2_gpu'*U2_gpu));
		U{1} = gather(U1_gpu);
		U31_gpu = gpuArray(kr(U{3},U{1}));
		U2_gpu = T2_gpu*U31_gpu*pinv((U3_gpu'*U3_gpu).*(U1_gpu'*U1_gpu));
		U{2} = gather(U2_gpu);
		U21_gpu = gpuArray(kr(U{2},U{1}));
		U3_gpu = T3_gpu*U21_gpu*pinv((U2_gpu'*U2_gpu).*(U1_gpu'*U1_gpu));
		U{3} = gather(U3_gpu);
	end
	t2=clock;
	time = etime(t2,t1);
	
	res = cpdgen(U);
	error = norm(res(:)-T(:))/norm(T(:));
	fid = fopen('cp_gpu.txt','a');
	fprintf(fid,'%d  %g  %fs\n', a, error , time);
	fclose(fid);
	clear all;

end
