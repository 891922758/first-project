
clear all;
addpath('/home/luhan/decomposition/CPU/matlab/tensorlab/');
for a = 100:100:1000
	size_tens = [a,a,a];
	r = 0.1*a;
	size_core = [a*0.1,a*0.1,a*0.1];
	[U0,S0] = lmlra_rnd(size_tens,size_core);
	S0 = single(rand(r,r,r));
	U0{1} = single(rand(a,r));
	U0{2} = single(rand(a,r));
	U0{3} = single(rand(a,r));
	T = lmlragen(U0,S0);
	T = single(T);

	t0=clock;
	t1 = tens2mat(T,1)*(tens2mat(T,1))';
	t2 = tens2mat(T,2)*(tens2mat(T,2))';
	t3 = tens2mat(T,3)*(tens2mat(T,3))';
	[U{1},~] = eig(t1);
	Uhat{1} = U{1}(1:a,a-r+1:a);
	[U{2},~] = eig(t2);
	Uhat{2} = U{2}(1:a,a-r+1:a);
	[U{3},~] = eig(t3);
	Uhat{3} = U{3}(1:a,a-r+1:a);
	S = tmprod(T,Uhat,1:3,'T');
	t4=clock;

	res = lmlragen(Uhat,S);
	error = norm(res(:)-T(:))/norm(T(:));
	fid = fopen('large_tucker.txt','a');
	fprintf(fid,'%d  %g  %fs\n', a, error , etime(t4,t0));
	fclose(fid);
	clear all;

end
