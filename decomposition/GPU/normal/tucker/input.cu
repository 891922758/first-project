/*************************************************************************
	> File Name: decom.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2019年04月23日 星期二 15时53分16秒
 ************************************************************************/

#include "head.h"

int main(int argc,char *argv[]){

for(int i =140;i<=170;i=i+10){
	clock_t t1,t2;
	double times=0.0;

	const long a = i;
	const long b = a;
	const long c = a;
	const long r1 = long(i*0.1);
	const long r2 = r1;
	const long r3 = r1;
	
	dt *X,*core,*U1,*U2,*U3;
	cudaHostAlloc((void**)&X,sizeof(dt)*a*b*c,0);
	cudaHostAlloc((void**)&core,sizeof(dt)*r1*r2*r3,0);
	cudaHostAlloc((void**)&U1,sizeof(dt)*a*r1,0);
	cudaHostAlloc((void**)&U2,sizeof(dt)*b*r2,0);
	cudaHostAlloc((void**)&U3,sizeof(dt)*c*r3,0);

	gentuTensor(X,a,b,c,r1,r2,r3);
	t1=clock();
	tucker_tensorcore(X,core,U1,U2,U3,a,b,c,r1,r2,r3);
	t2=clock();
	times = (double)(t2-t1)/CLOCKS_PER_SEC;
	cout<<i<<" "<<times<<"s"<<endl;
	ofstream outfile("tuckertensorcore.txt",ios::app);
	outfile<<i<<"   "<<times<<"s"<<endl;
	outfile.close();

	cudaFreeHost(X);
	cudaFreeHost(core);
	cudaFreeHost(U1);
	cudaFreeHost(U2);
	cudaFreeHost(U3);
}
	return 0;
}

