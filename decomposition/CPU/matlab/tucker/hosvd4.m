
clear all;
addpath('/home/luhan/decomposition/CPU/matlab/tensorlab/');
for a = 100:40:300
	size_tens = [a,a,a,a];
	r = 0.1*a;
	size_core = [a*0.1,a*0.1,a*0.1,a*0.1];
	[U0,S0] = lmlra_rnd(size_tens,size_core);
	S0 = single(rand(r,r,r,r));
	U0{1} = single(rand(a,r));
	U0{2} = single(rand(a,r));
	U0{3} = single(rand(a,r));
	U0{4} = single(rand(a,r));
	T = lmlragen(U0,S0);

	time0=clock;
	[U,S]=mlsvd(T,size_core,[],'LargeScale',true);
%	[U,S]=mlsvd(T,size_core);
	time1=clock;

	res = lmlragen(U,S);
	error = norm(res(:)-T(:))/norm(T(:));
	fid = fopen('hosvd4.txt','a');
	fprintf(fid,'%d  %g  %fs\n', a, error , etime(time1,time0));
	fclose(fid);
end

