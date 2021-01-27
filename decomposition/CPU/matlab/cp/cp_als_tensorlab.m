
clear all;
addpath('/home/luhan/decomposition/CPU/matlab/tensorlab/');
for a = 100:10:1200
	size_tens = [a,a,a];
	r = a*0.1;
	U0 = cpd_rnd(size_tens, a*0.1);
	U0{1} = single(rand(a,r));
	U0{2} = single(rand(a,r));
	U0{3} = single(rand(a,r));
	T = cpdgen(U0);

	U = cpd_rnd(size_tens, a*0.1);
	U{1} = single(rand(a,r));
	U{2} = single(rand(a,r));
	U{3} = single(rand(a,r));

	t1=clock;
	Uhat = cpd_als(T,U);
	t2 = clock;
	time = etime(t2,t1);
	Uhat{1} = single(Uhat{1});
	Uhat{2} = single(Uhat{2});
	Uhat{3} = single(Uhat{3});
	error = frobcpdres(T,Uhat)/frob(T);
	fid = fopen('cp_als_tensorlab.txt','a');
	fprintf(fid,'%d  %g  %fs\n', a, error , time);
	fclose(fid);
	clear all;

end
