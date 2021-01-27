
clear all;
addpath('/home/luhan/decomposition/CPU/matlab/tensorlab/');

for a = 160:80:2000
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
	for i = 1:10
		U{1} = tens2mat(T,1)*kr(U{3},U{2})*pinv((U{3}'*U{3}).*(U{2}'*U{2}));
		U{2} = tens2mat(T,2)*kr(U{3},U{1})*pinv((U{3}'*U{3}).*(U{1}'*U{1}));
		U{3} = tens2mat(T,3)*kr(U{2},U{1})*pinv((U{2}'*U{2}).*(U{1}'*U{1}));
	end									
	t2=clock;
	time = etime(t2,t1);	
	error = frobcpdres(T,U)/frob(T);

	fid = fopen('large_cp.txt','a');
	fprintf(fid,'%d  %g  %fs\n', a, error , time);
	fclose(fid);
	clear all;
end

