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
for(int i =80;i<=180;i=i+10){

	const long a = i;
	const long b = a;
	const long c = a;
	const long d = a;
	const long r = (int)(a*0.1);
	dt *X,*A,*B,*C,*D;
	cudaHostAlloc((void**)&X,sizeof(dt)*a*b*c*d,0);
	cudaHostAlloc((void**)&A,sizeof(dt)*a*r,0);
	cudaHostAlloc((void**)&B,sizeof(dt)*b*r,0);
	cudaHostAlloc((void**)&C,sizeof(dt)*c*r,0);
	cudaHostAlloc((void**)&D,sizeof(dt)*d*r,0);

	gencpTensor4(X,a,b,c,d,r);
/*	for(int i = 0;i<a*b*c*d;i++){
		X[i]=rand()/(RAND_MAX*0.1);
	}
*/
	t1=clock();
	cp_als4(X,A,B,C,D,a,b,c,d,r);
	t2=clock();

	cout<<i<<" "<<(double)(t2-t1)/CLOCKS_PER_SEC<<"s"<<endl;

	ofstream outfile("high-cp.txt",ios::app);
	outfile<<i<<"   "<<(double)(t2-t1)/CLOCKS_PER_SEC<<"s"<<endl;
	outfile.close();

	cudaFreeHost(X);
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFreeHost(D);

}
	return 0;
}

