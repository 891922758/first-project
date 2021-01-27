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

	dt A[a*r1]={1,3,2,0,-1,2};
	dt B[b*r2]={2,0,4,6,-1,1,1,2};
	dt C[c*r3]={3,-1,0,2};
	dt G[r1*r2*r3]={-2,-1,0,3,4,1,2,3};
	dt *X;
	cudaHostAlloc((void**)&X,sizeof(dt)*a*b*c,0);
	dt *core,*U1,*U2,*U3;
	cudaHostAlloc((void**)&U1,sizeof(dt)*a*r1,0);
	cudaHostAlloc((void**)&U2,sizeof(dt)*b*r2,0);
	cudaHostAlloc((void**)&U3,sizeof(dt)*c*r3,0);
	cudaHostAlloc((void**)&core,sizeof(dt)*r1*r2*r3,0);
	gentuTensor1(X,A,B,C,G,a,b,c,r1,r2,r3);
*/
	clock_t t1,t2;
for(int i = 180;i<=190;i=i+20){
	const long a = i;
	const long b = a;
	const long c = a;
	const long d = a;
	const long r1 = int(a*0.1);
	const long r2 = r1;
	const long r3 = r1;
	const long r4 = r1;
	dt *X,*core,*U1,*U2,*U3,*U4;
	cudaHostAlloc((void**)&X,sizeof(dt)*a*b*c*d,0);
	cudaHostAlloc((void**)&core,sizeof(dt)*r1*r2*r3*r4,0);
	cudaHostAlloc((void**)&U1,sizeof(dt)*a*r1,0);
	cudaHostAlloc((void**)&U2,sizeof(dt)*b*r2,0);
	cudaHostAlloc((void**)&U3,sizeof(dt)*c*r3,0);
	cudaHostAlloc((void**)&U4,sizeof(dt)*d*r4,0);
	gentuTensor4(X,a,b,c,d,r1,r2,r3,r4);
	
	cout<<i<<" "<<r1<<endl;
	t1=clock();
	tucker_hosvd4(X,core,U1,U2,U3,U4,a,b,c,d,r1,r2,r3,r4);
	t2=clock();
	cout<<(double)(t2-t1)/CLOCKS_PER_SEC<<"s"<<endl;
	ofstream outfile("high-tucker.txt",ios::app);
	outfile<<i<<"   "<<(double)(t2-t1)/CLOCKS_PER_SEC<<"s"<<endl;
	outfile.close();

	cudaFreeHost(X);
	cudaFreeHost(core);
	cudaFreeHost(U1);
	cudaFreeHost(U2);
	cudaFreeHost(U3);
	cudaFreeHost(U4);
}
	return 0;
}

