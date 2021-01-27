
clear all

for i=200:10:300
	a=i;
	b=a;
	c=a;
	d=a;
	A=rand(a,b);
	B=rand(c,d);
	C=rand(a*c,b*d);
	
	t1=clock;
	for r = 1:b
		for l = 1:d
			temp=B(:,l)*A(:,r)';
                    	C(:,(r-1)*d+l) =temp(:);
		end
	end
	t2=clock;
	time=etime(t2,t1);
	display(a);
	display(time);
	fid = fopen('tp.txt','a');
	fprintf(fid,'%d %fs\n',a,time);
	fclose(fid);

end
