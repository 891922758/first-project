#include "cudecompose.h"

typedef float dt;
using namespace std;

void printTensor(dt *A,int a,int b,int c){
	for(int i = 0;i<c;i++){
		for(int j = 0;j<a;j++){
			for(int k =0;k<b;k++){
				cout<<A[i*a*b+k*a+j]<<"  ";
			}
			cout<<endl;
		}
		cout<<"-----------------------------------"<<endl;
	}
	cout<<endl;
}

int main(int argc,char *argv[]){
	long long a = 3;
	long long b = 3;
	long long r = 2;	//a*r b*r
	dt *A,*B;
	cudaHostAlloc((void**)&A,sizeof(dt)*a*r,0);
	cudaHostAlloc((void**)&B,sizeof(dt)*b*r,0);
	for(long long i = 0;i<a*r;i++){
		A[i] = rand()*0.1/(RAND_MAX);		//initial Tensor A
	}
	for(long long i = 0;i<b*r;i++){
		B[i] = rand()*0.1/(RAND_MAX);		//initial Tensor A
	}
printTensor(A,a,r,1);
printTensor(B,b,r,1);
	dt *AkrB;
	cudaHostAlloc((void**)&AkrB,sizeof(dt)*a*b*r,0);

	dt *d_A;
	dt *d_B;
	dt *d_AkrB;
	cudaMalloc((void **)&d_A,sizeof(dt)*a*r);
	cudaMalloc((void **)&d_B,sizeof(dt)*b*r);
	cudaMalloc((void **)&d_AkrB,sizeof(dt)*a*b*r);
	cudaMemcpyAsync(d_A,A,sizeof(dt)*a*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_B,B,sizeof(dt)*b*r,cudaMemcpyHostToDevice,0);
	cudaDeviceSynchronize();

	kr(d_A,d_B,d_AkrB,a,b,r);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaMemcpyAsync(AkrB,d_AkrB,sizeof(dt)*a*b*r,cudaMemcpyDeviceToHost,0);
	cudaDeviceSynchronize();
	cudaFree(d_AkrB);

printTensor(AkrB,a*b,r,1);
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(AkrB);
	return 0;
}


