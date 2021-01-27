
for i = 100:100:2000
	a=i;
	b=i;
	A=rand(a,b);
	B=rand(a,b);
	C=rand(a,b);
	t1=clock;
	for k = 1:3
		C=A.*B;
	end
	t2=clock();
	time=(etime(t2,t1)/3);
	fid = fopen('hd.txt','a');
	fprintf(fid,'%d %fs\n',a,time);
	fclose(fid);

end
