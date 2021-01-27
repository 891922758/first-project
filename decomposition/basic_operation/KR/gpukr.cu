#include <iostream>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <time.h>
 
typedef float dt;
using namespace std;

__global__ void krpro(dt *M,dt *N,dt *res,long long m,long long n,long long r){
	//m*r and n*r to (m*n)*r	
	long long i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<m*n*r){
		long long row = i%(m*n);
		long long col = i/(m*n);
		res[col*m*n+row] = M[(row/n)+col*m]*N[(row%n)+col*n];
	}
    __syncthreads();
}

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
for(int hh = 160;hh<=1600;hh=hh+160){
	long long a = hh;
	long long b = a;
	long long r = a;	//a*r b*r
	dt *A,*B;
	cout<<a<<endl;
	cudaHostAlloc((void**)&A,sizeof(dt)*a*r,0);
	cudaHostAlloc((void**)&B,sizeof(dt)*b*r,0);
	srand(5);
	for(long long i = 0;i<a*r;i++){
		A[i] = rand()*0.1/(RAND_MAX*0.1);		//initial Tensor A
	}
	for(long long i = 0;i<b*r;i++){
		B[i] = rand()*0.1/(RAND_MAX*0.1);		//initial Tensor A
	}
//printTensor(A,a,r,1);
//printTensor(B,b,r,1);
	dt *AkrB;
	cudaHostAlloc((void**)&AkrB,sizeof(dt)*a*b*r,0);

	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	dim3 threads(512,1,1);
	dim3 blocks((a*b*r+512-1)/512,1,1);
//	dim3 blocks(1024,1,1);
//	cout<<(a*b*r+1023)/1024<<endl;
	dt *d_A;
	dt *d_B;
	dt *d_AkrB;
	cudaMalloc((void **)&d_A,sizeof(dt)*a*r);
	cudaMalloc((void **)&d_B,sizeof(dt)*b*r);
	cudaMalloc((void **)&d_AkrB,sizeof(dt)*a*b*r);
	cudaMemcpyAsync(d_A,A,sizeof(dt)*a*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_B,B,sizeof(dt)*b*r,cudaMemcpyHostToDevice,0);
	cudaDeviceSynchronize();

//warm up
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,a,1,&alpha,d_B,b,b,d_A,a,a,&beta,d_AkrB,b,b*a,r);
	krpro<<<blocks,threads>>>(d_A,d_B,d_AkrB,a,b,r);

	cudaEvent_t start,stop;
	dt elapsedTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
for(int j = 0;j<10;j++){
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,a,1,&alpha,d_B,b,b,d_A,a,a,&beta,d_AkrB,b,b*a,r);
//	krpro<<<blocks,threads>>>(d_A,d_B,d_AkrB,a,b,r);
}
	cudaEventRecord(stop,0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,start,stop);
	cout<<elapsedTime/10<<endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaMemcpyAsync(AkrB,d_AkrB,sizeof(dt)*a*b*r,cudaMemcpyDeviceToHost,0);
	cudaDeviceSynchronize();
	cudaFree(d_AkrB);
	cublasDestroy(handle);	

//printTensor(AkrB,a*b,r,1);
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(AkrB);
}
	return 0;
}


