
clear all;
addpath('/home/luhan/decomposition/CPU/matlab/tensorlab/');
for a = 1280:80:2000
	size_tens = [a,a,a];
	r = 0.1*a;
	size_core = [a*0.1,a*0.1,a*0.1];
	[U0,S0] = lmlra_rnd(size_tens,size_core);
	S0 = single(rand(r,r,r));
	U0{1} = single(rand(a,r));
	U0{2} = single(rand(a,r));
	U0{3} = single(rand(a,r));
	T = lmlragen(U0,S0);

	t0=clock;
	[U,S]=mlsvd(T,size_core,[],'LargeScale',true);
	t1=clock;

	res = lmlragen(U,S);
	error = norm(res(:)-T(:))/norm(T(:));
	fid = fopen('hosvd_tensorlab.txt','a');
	fprintf(fid,'%d  %g  %fs\n', a, error , etime(t1,t0));
	fclose(fid);
	clear all;

end

