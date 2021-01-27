/*************************************************************************
	> File Name: decom.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2019年04月23日 星期二 15时53分16秒
 ************************************************************************/

#include "head.h"

int main(int argc,char *argv[]){
/*	const int a=3;
	const int b=4;
	const int c=2;
	const int r1=2;
	const int r2=2;
	const int r3=2;

	dt X[a*b*c]={2,0,4,6,-1,1,1,2,6,3,8,2,12,4,8,22,21,1,3,4,2,6,4,2};
	dt *core,*U1,*U2,*U3;
	cudaHostAlloc((void**)&U1,sizeof(dt)*a*r1,0);
	cudaHostAlloc((void**)&U2,sizeof(dt)*b*r2,0);
	cudaHostAlloc((void**)&U3,sizeof(dt)*c*r3,0);
	cudaHostAlloc((void**)&core,sizeof(dt)*r1*r2*r3,0);
	
	tucker_hosvd(X,core,U1,U2,U3,a,b,c,r1,r2,r3);

	cudaFreeHost(U1);
	cudaFreeHost(U2);
	cudaFreeHost(U3);
	cudaFreeHost(core);
*/
	clock_t t1,t2;
for(int i = 1280;i<=2000;i=i+80){
	cout<<i<<endl;
	const long a = i;
	const long b = a;
	const long c = a;
	const long r1 = int(a*0.1);
	const long r2 = r1;
	const long r3 = r1;
	dt *X,*core,*U1,*U2,*U3;
	cudaHostAlloc((void**)&X,sizeof(dt)*a*b*c,0);
	for(long i = 0;i<a*b*c;i++ ){
		X[i] = rand()*0.1/(RAND_MAX);
	}
	cudaHostAlloc((void**)&core,sizeof(dt)*r1*r2*r3,0);
	cudaHostAlloc((void**)&U1,sizeof(dt)*a*r1,0);
	cudaHostAlloc((void**)&U2,sizeof(dt)*b*r2,0);
	cudaHostAlloc((void**)&U3,sizeof(dt)*c*r3,0);

//	gentuTensor(X,a,b,c,r1,r2,r3);
	
	t1=clock();
	tucker_hosvd(X,core,U1,U2,U3,a,b,c,r1,r2,r3);
	t2=clock();
	cout<<(double)(t2-t1)/CLOCKS_PER_SEC<<"s"<<endl;
	ofstream outfile("la-tucker.txt",ios::app);
	outfile<<i<<"  "<<(double)(t2-t1)/CLOCKS_PER_SEC<<"s"<<endl;
	outfile.close();

	cudaFreeHost(X);
	cudaFreeHost(core);
	cudaFreeHost(U1);
	cudaFreeHost(U2);
	cudaFreeHost(U3);
}
	return 0;
}

