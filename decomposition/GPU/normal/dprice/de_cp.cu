/*************************************************************************
	> File Name: decom.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2019年04月23日 星期二 15时53分16秒
 ************************************************************************/

#include "head.h"

int main(int argc,char *argv[]){
/*	const int a=3;
	const int b=2;
	const int c=2;
	const int r=2;

	dt X[a*b*c]={0,1,2,3,4,5,6,7,8,9,10,11};
	dt B[b*r]={2,1,6,3};	//
	dt C[c*r]={0,7,1,2};
	dt A[a*r]={0,0,0,0,0,0};
*/
	const int a = 800;
	const int b = 800;
	const int c = 800;
	const int r = 80;
	dt *X,*A,*B,*C;
	cudaHostAlloc((void**)&X,sizeof(dt)*a*b*c,0);
	cudaHostAlloc((void**)&A,sizeof(dt)*a*r,0);
	cudaHostAlloc((void**)&B,sizeof(dt)*b*r,0);
	cudaHostAlloc((void**)&C,sizeof(dt)*c*r,0);
	gencpTensor(X,a,b,c,r);
	srand((unsigned)time(NULL));

//	for(int i = 0;i<a*b*c;i++){
//		X[i]=rand()*0.1/RAND_MAX;
//	}
	for(int i = 0;i<b*r;i++){
		B[i]=rand()*0.1/(RAND_MAX*0.1);
	}
	for(int i = 0;i<c*r;i++){
		C[i]=rand()*0.1/(RAND_MAX*0.1);
	}
	cp_als(X,A,B,C,a,b,c,r);

	cudaFreeHost(X);
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);

	return 0;
}

