/*************************************************************************
	> File Name: decom.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2019年04月23日 星期二 15时53分16秒
 ************************************************************************/

#include "head.h"

int main(int argc,char *argv[]){
	const int a=3;
	const int b=2;
	const int c=2;
	const int d=2;
	const int r=2;

	dt X[a*b*c*d]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23};
	dt B[b*r]={2,1,6,3};	//
	dt C[c*r]={0,7,1,2};
	dt D[d*r]={1,3,0,-1};
	dt A[a*r]={0,0,0,0,0,0};
	cp_als4(X,A,B,C,D,a,b,c,d,r);

/*
	clock_t t1,t2;
//for(int i =50;i<=1000;i=i+50){

	const int a = 50;
	const int b = a;
	const int c = a;
	const int d = a;
	const int r = (int)(a*0.1);
	dt *X,*A,*B,*C,*D;
	cudaHostAlloc((void**)&X,sizeof(dt)*a*b*c*d,0);
	cudaHostAlloc((void**)&A,sizeof(dt)*a*r,0);
	cudaHostAlloc((void**)&B,sizeof(dt)*b*r,0);
	cudaHostAlloc((void**)&C,sizeof(dt)*c*r,0);
	cudaHostAlloc((void**)&D,sizeof(dt)*d*r,0);
	gencpTensor4(X,a,b,c,d,r);
	//cout<<i<<endl;
	t1=clock();
	cp_als4(X,A,B,C,D,a,b,c,d,r);
	t2=clock();
//	cout<<(double)(t2-t1)/CLOCKS_PER_SEC<<"s"<<endl;
	cudaFreeHost(X);
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFreeHost(D);

//}

*/
	return 0;
}

