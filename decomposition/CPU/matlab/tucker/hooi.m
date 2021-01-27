

clear all;
addpath('/home/luhan/tensorlab/');
for a = 50:50:1200
	size_tens = [a,a,a];
	size_core = [a*0.1,a*0.1,a*0.1];
	[U0,S0] = lmlra_rnd(size_tens,size_core);
	T = lmlragen(U0,S0);
	t1=clock;
	[U{1},~,~] = svds(tens2mat(T,1),size_core(1));
	[U{2},~,~] = svds(tens2mat(T,2),size_core(2));
	[U{3},~,~] = svds(tens2mat(T,3),size_core(3));
	N=3;
	for i = 1:100
		for n = 1:N
			mode = [1:n-1 n+1:N];
			[Y] = tmprod(T,U(mode),mode,'T');
			[U{n},~,~] = svds(tens2mat(Y,n),size_core(n));
		end
		S = tmprod(T,U,1:3,'T');
	end
	t2=clock;
	time = etime(t2,t1);
	res = lmlragen(U,S);
	error = norm(res(:)-T(:))/norm(T(:));
	fid = fopen('hooi.txt','a');
	fprintf(fid,'%d  %g  %fs\n', a,error , time);
	fclose(fid);
end

