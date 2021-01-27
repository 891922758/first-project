#include <cuda_fp16.h>
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

__global__  void floattohalf(dt *AA,half *BB,long m){
	long i = blockIdx.x*blockDim.x+threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	if(i<m){
		BB[i]=__float2half(AA[i]);
		i+=temp;
	}
	__syncthreads();
}

void f2h(dt *A,half *B,long num){
	dim3 threads(512,1,1);
	dim3 blocks((num+512-1)/512,1,1);	
	floattohalf<<<blocks,threads>>>(A,B,num);
}

int main(int argc, char *argv[]){
	cudaSetDevice(1);

	for(int hh = 1000;hh<=2000;hh=hh+2000){
//		clock_t t1,t2;
		long long a = hh;
		long long r = hh;
		long long b = hh;	//a*r b*r
		dt *A,*B,*C;
		cudaHostAlloc((void**)&A,sizeof(dt)*a*r,0);
		cudaHostAlloc((void**)&B,sizeof(dt)*b*r,0);
		cudaHostAlloc((void**)&C,sizeof(dt)*b*a,0);
		srand(5);
		for(long long i = 0;i<a*r;i++){
			A[i] = rand()*0.1/(RAND_MAX*0.1);		//initial Tensor A
		}	
		for(long long i = 0;i<b*r;i++){
			B[i] = rand()*0.1/(RAND_MAX*0.1);		//initial Tensor A
		}

//		printTensor(A,a,r,1);
//		printTensor(B,r,b,1);
		
		dt alpha = 1.0;
		dt beta = 0.0;
		cublasHandle_t handle;
		cublasCreate(&handle);
		//cublasMath_t CUBLAS_TENSOR_OP_MATH;
		cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);
		
		dt *d_A;
		dt *d_B;
		dt *d_C;
		cudaMalloc((void **)&d_A,sizeof(dt)*a*r);
		cudaMalloc((void **)&d_B,sizeof(dt)*b*r);
		cudaMalloc((void **)&d_C,sizeof(dt)*a*b);
		cudaMemcpyAsync(d_A,A,sizeof(dt)*a*r,cudaMemcpyHostToDevice,0);
		cudaMemcpyAsync(d_B,B,sizeof(dt)*b*r,cudaMemcpyHostToDevice,0);
		cudaDeviceSynchronize();
	
		half *h_A,*h_B;
		cudaMalloc((void **)&h_A,sizeof(half)*a*r);
		cudaMalloc((void **)&h_B,sizeof(half)*b*r);

		cudaEvent_t start,stop;
		dt elapsedTime = 0.0;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		
	//	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,a,b,r,&alpha,d_A,a,d_B,r,&beta,d_C,a);
		
		f2h(d_A,h_A,a*r);
		f2h(d_B,h_B,b*r);
//		cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,a,b,r,&alpha,h_A,CUDA_R_16F,a,h_B,CUDA_R_16F,r,&beta,d_C,CUDA_R_32F,a,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	
		cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,a,b,r,&alpha,d_A,CUDA_R_32F,a,d_B,CUDA_R_32F,r,&beta,d_C,CUDA_R_32F,a,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		cudaEventRecord(stop,0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime,start,stop);
		cout<<elapsedTime/1<<endl;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		cudaMemcpyAsync(C,d_C,sizeof(dt)*a*b,cudaMemcpyDeviceToHost,0);
		cudaDeviceSynchronize();
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
		cudaFree(h_A);
		cudaFree(h_B);
		cublasDestroy(handle);	
		
//		printTensor(C,a,b,1);

		cudaFreeHost(A);
		cudaFreeHost(B);
		cudaFreeHost(C);
	}

	return 0;
}




