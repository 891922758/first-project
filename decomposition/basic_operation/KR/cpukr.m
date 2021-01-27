close all;
clear all;
addpath('/home/luhan/tensorlab');
for i=1900:100:2000
	a=i;
	b=a;
	r=a;
	A=rand(a,r);
	B=rand(b,r);
	AkrB=rand(a*b,r);
	t1=clock;
	for k = 1:2
		for j = 1:r
			temp = B(:,j)*A(:,j)';
			AkrB(:,j)=temp(:);
		end
	end
	t2=clock;
	time=(etime(t2,t1)/2);
	fid = fopen('luhan.txt','a');
	fprintf(fid,'%d %fs\n',a,time);
	fclose(fid);
end
