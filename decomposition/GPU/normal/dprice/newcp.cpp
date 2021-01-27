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
	clock_t t1,t2;
	double times=0.0;
for(int i =100;i<=1200;i=i+100){

	const int a = i;
	const int b = a;
	const int c = a;
	const int r = (int)(a*0.1);
	dt *X,*A,*B,*C;
	cudaHostAlloc((void**)&X,sizeof(dt)*a*b*c,0);
	cudaHostAlloc((void**)&A,sizeof(dt)*a*r,0);
	cudaHostAlloc((void**)&B,sizeof(dt)*b*r,0);
	cudaHostAlloc((void**)&C,sizeof(dt)*c*r,0);
	gencpTensor(X,a,b,c,r);
	cout<<i<<endl;

	t1=clock();
	cp_als(X,A,B,C,a,b,c,r);
	t2=clock();
	times = (double)(t2-t1)/CLOCKS_PER_SEC;
	cout<<times<<"s"<<endl;

	cudaFreeHost(X);
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);

}
	return 0;
}

