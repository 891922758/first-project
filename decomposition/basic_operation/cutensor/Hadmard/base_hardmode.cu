
#include <iostream>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>
 
typedef float dt;
using namespace std;

__global__ void hardm(dt *M,dt *N,dt *res,long  m, long n){
	 long i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<m*n){
		res[i] = M[i]*N[i];
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

	for(long hh = 160;hh<=1600;hh=hh+160){
		long a = hh;
		long b = a;
		dt *A,*B,*C;
		cout<<a<<endl;
		cudaHostAlloc((void**)&A,sizeof(dt)*a*b,0);
		cudaHostAlloc((void**)&B,sizeof(dt)*a*b,0);
		cudaHostAlloc((void**)&C,sizeof(dt)*a*b,0);
		srand(2);
		for(long i = 0;i<a*b;i++){
			A[i] = rand()/(RAND_MAX)*100;		//initial Tensor A
		}
		for(long i = 0;i<a*b;i++){
			B[i] = rand()/(RAND_MAX)*100;		//initial Tensor A
		}
		for(long i = 0;i<a*b;i++){
			C[i] = rand()/(RAND_MAX)*100;		//initial Tensor A
		}
//printTensor(A,a,b,1);
//printTensor(B,a,b,1);
		dt alpha = 1.0;
		dt beta = 0.0;
		cublasHandle_t handle;
		cublasCreate(&handle);
		dim3 threads(1024,1,1);
		dim3 blocks((a*b+1024-1)/1024,1,1);
		dt *d_A;
		dt *d_B;
		dt *d_C;
		cudaMalloc((void **)&d_A,sizeof(dt)*a*b);
		cudaMalloc((void **)&d_B,sizeof(dt)*b*b);
		cudaMalloc((void **)&d_C,sizeof(dt)*a*b);
		cudaMemcpyAsync(d_A,A,sizeof(dt)*a*b,cudaMemcpyHostToDevice,0);
		cudaMemcpyAsync(d_B,B,sizeof(dt)*b*a,cudaMemcpyHostToDevice,0);
		cudaDeviceSynchronize();
	//warm	
		cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,1,1,&alpha,d_A,1,1,d_B,1,1,&beta,d_C,1,1,a*b);
		cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,1,1,&alpha,d_A,1,1,d_B,1,1,&beta,d_C,1,1,a*b);
		cudaDeviceSynchronize();

		cudaEvent_t start,stop;
		dt elapsedTime = 0.0;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
	for(int j = 0;j<10;j++){
		hardm<<<blocks,threads>>>(d_A,d_B,d_C,a,b);
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
		cudaMemcpyAsync(C,d_C,sizeof(dt)*a*b,cudaMemcpyDeviceToHost,0);
		cudaDeviceSynchronize();
//printTensor(C,a,b,1);
		cudaFree(d_C);
		cublasDestroy(handle);	

		cudaFreeHost(A);
		cudaFreeHost(B);
		cudaFreeHost(C);
	
	}
	
	return 0;
}
