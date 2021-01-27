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
	const int r=2;

	dt X[a*b*c]={2,0,4,6,-1,1,1,2,6,3,8,2,12,4,8,22,21,1,3,4,2,6,4,2};
	dt A[a*r]={3,2,12,5,0,3};
	dt B[b*r]={7,2,8,9,13,2,12,11};
	dt C[c*r]={4,5,8,2};
	cp_als(X,A,B,C,a,b,c,r);
*/
	clock_t t1,t2;
 for(int i =1280;i<=2000;i=i+80){

	long a = i;
	long b = a;
	long c = a;
	long r =(long)(a*0.1);
	cout<<a<<endl;

	dt *A,*B,*C;
	dt *X ;
	cudaHostAlloc((void**)&A,sizeof(dt)*a*r,0);
	cudaHostAlloc((void**)&B,sizeof(dt)*b*r,0);
	cudaHostAlloc((void**)&C,sizeof(dt)*c*r,0);
	cudaHostAlloc((void**)&X,sizeof(dt)*a*b*c,0);
	for(long i = 0;i<a*b*c;i++ ){
		X[i] = rand()*0.1/(RAND_MAX);
	}

//	gencpTensor(X,a,b,c,r);
	t1=clock();
	cp_als(X,A,B,C,a,b,c,r);
	t2=clock();
	cout<<(double)(t2-t1)/CLOCKS_PER_SEC<<"s"<<endl;
	ofstream outfile("large-cp.txt",ios::app);
	outfile<<i<<"  "<<(double)(t2-t1)/CLOCKS_PER_SEC<<"s"<<endl;
	outfile.close();

	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFreeHost(X);

	}

	return 0;
}

