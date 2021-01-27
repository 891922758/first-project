#include <iostream>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "GPUTimer.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>

using namespace std;
typedef float dt;

__global__  void floattohalf(dt *AA,half *BB,long m){
	long i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<m){
		BB[i]=__float2half(AA[i]);
	}
	__syncthreads();
}

void f2h(dt *A,half *B,long num){
	dim3 threads(512,1,1);
	dim3 blocks((num+512-1)/512,1,1);	
	floattohalf<<<blocks,threads>>>(A,B,num);
}

__global__ void krpro(dt *M,dt *N,dt *res,long long m,long long n,long long r){
	//m*r and n*r to (m*n)*r	
	long long i = blockIdx.x*blockDim.x+threadIdx.x;
	const long long temp = blockDim.x*gridDim.x;
	while(i<m*n*r){
		long long row = i%(m*n);
		long long col = i/(m*n);
		res[col*m*n+row] = M[(row/n)+col*m]*N[(row%n)+col*n];
		i+=temp;
	}
    __syncthreads();
}
__global__ void tensorToMode1(dt *T1,dt *T2,int m,int n,int k ){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<m*n*k){
		int tube = i/(m*n);
		int row = (i-tube*(m*n))%m;
		int col = (i-tube*(m*n))/m;
		T2[tube*m*n+col*m+row] = T1[tube*m*n+col*m+row];
	}
	__syncthreads();
	
}

void printTensor(dt *A,int a,int b,int c){
	dt *h_A;
	cudaHostAlloc((void**)&h_A,sizeof(dt)*a*b*c,0);
	cudaMemcpyAsync(h_A,A,sizeof(dt)*a*b*c,cudaMemcpyDeviceToHost,0);
	cudaDeviceSynchronize();

	for(int i = 0;i<c;i++){
		for(int j = 0;j<a;j++){
			for(int k =0;k<b;k++){
				cout<<h_A[i*a*b+k*a+j]<<"  ";
			}
			cout<<endl;
		}
		cout<<"-----------------------------------"<<endl;
	}
	cout<<endl;
	cudaFreeHost(h_A);
}

int main(int argc,char *argv[]){
for(int i = 160;i<1400;i=i+160){
	int m = i;
	int n = m;
	int k = m;
	int r = (int)(m*0.1);
//	int r = m;
	cout<<m<<endl;
	size_t size=sizeof(dt);
	float time1 = 0.0;
	float time2 = 0.0;
	float time3 = 0.0;
	float temp = 0.0;
	dt *X,*B,*C;
	cudaHostAlloc((void**)&X,size*m*n*k,0);
	cudaHostAlloc((void**)&B,size*n*r,0);
	cudaHostAlloc((void**)&C,size*k*r,0);
//	srand(2);
	for(int i = 0;i<m*n*k;i++){
		X[i] = (((float) rand())/RAND_MAX - 0.5);
	}
	for(int i = 0;i<n*r;i++){
		B[i] = (((float) rand())/RAND_MAX - 0.5);
	}
	for(int i = 0;i<k*r;i++){
		C[i] = (((float) rand())/RAND_MAX - 0.5);
	}
	
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	GPUTimer timer;
	
	dt *d_X;
	cudaMalloc((void**)&d_X,size*m*n*k);
	cudaMemcpyAsync(d_X,X,size*m*n*k,cudaMemcpyHostToDevice,0);
	dt *d_CkrB,*d_C,*d_B,*d_result;
	cudaMalloc((void**)&d_B,size*n*r);
	cudaMalloc((void**)&d_C,size*k*r);
	cudaMalloc((void**)&d_result,size*m*r);
	cudaMemcpyAsync(d_C,C,size*k*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_B,B,size*n*r,cudaMemcpyHostToDevice,0);
	cudaDeviceSynchronize();
	cudaMalloc((void**)&d_CkrB,size*n*k*r);
	half *h_CkrB,*h_X;
	cudaMalloc((void **)&h_X,sizeof(half)*m*n*k);
	cudaMalloc((void **)&h_CkrB,sizeof(half)*n*k*r);

	dim3 thread(512,1,1);
	dim3 block((m*n*k+512-1)/512,1,1); //for tensor matrix
	dim3 block1((n*k*r+512-1)/512,1,1); //for kr

//	cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);
	cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH);
	int L = 10;

for(int iter = 0;iter<L;++iter){
	//warm up
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,n,k,1,&alpha,d_B,n,n,d_C,k,k,&beta,d_CkrB,n,k*n,r);
	cudaDeviceSynchronize();
//	cout<<"d_CkrB"<<endl;
//	printTensor(d_CkrB,k*n,r,1);

	dt *d_X1;
	cudaMalloc((void**)&d_X1,size*m*n*k);

	timer.start();
	krpro<<<block1,thread>>>(d_C,d_B,d_CkrB,k,n,r);
	cudaDeviceSynchronize();
//	cout<<"unopt-d_CkrB"<<endl;
//	printTensor(d_CkrB,k*n,r,1);
	tensorToMode1<<<block,thread>>>(d_X,d_X1,m,n,k);
	cudaDeviceSynchronize();
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,r,n*k,&alpha,d_X1,m,d_CkrB,n*k,&beta,d_result,m);
	cudaDeviceSynchronize();
	time1 = time1+timer.seconds();
//	cout<<"unopt-d_result"<<endl;
//	printTensor(d_result+2*m,3,4,1);
//	printTensor(d_result,m,r,1);
	cudaFree(d_X1);
	
	timer.start();
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,n,k,1,&alpha,d_B,n,n,d_C,k,k,&beta,d_CkrB,n,k*n,r);
	cudaDeviceSynchronize();
//	cout<<"opt-d_CkrB"<<endl;
//	printTensor(d_CkrB,k*n,r,1);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,r,n*k,&alpha,d_X,m,d_CkrB,n*k,&beta,d_result,m);
	cudaDeviceSynchronize();
	time2 =time2+timer.seconds();
//	cout<<"opt-d_result"<<endl;
//	printTensor(d_result+2*m,3,4,1);
//	printTensor(d_result,m,r,1);

	timer.start();
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,n,k,1,&alpha,d_B,n,n,d_C,k,k,&beta,d_CkrB,n,k*n,r);
	cudaDeviceSynchronize();
	temp = timer.seconds();
//	cout<<"opt-d_CkrB"<<endl;
//	printTensor(d_CkrB,k*n,r,1);
	f2h(d_CkrB,h_CkrB,k*n*r);
	f2h(d_X,h_X,m*n*k);
	cudaDeviceSynchronize();
	timer.start();
	cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,r,n*k,&alpha,h_X,CUDA_R_16F,m,h_CkrB,CUDA_R_16F,n*k,&beta,d_result,CUDA_R_32F,m,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//	cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,r,n*k,&alpha,h_X,CUDA_R_16F,m,h_CkrB,CUDA_R_16F,n*k,&beta,d_result,CUDA_R_32F,m,CUDA_R_32F,CUBLAS_GEMM_DEFAULT);
	cudaDeviceSynchronize();
	time3 =time3+ temp+timer.seconds();
//	cout<<"opt-d_result"<<endl;
//	printTensor(d_result+2*m,3,4,1);
//	printTensor(d_result,m,r,1);

	if(iter == L-1){
		cout<<"mode1-unop = "<<time1/L<<"ms"<<endl;
		cout<<"mode1-op = "<<time2/L<<"ms"<<endl;
		cout<<"mode1-op-tensor-core = "<<time3/L<<"ms"<<endl;
	}
}

	cublasDestroy(handle);	

	cudaFreeHost(X);
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFree(h_X);
	cudaFree(h_CkrB);
	cudaFree(d_X);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_CkrB);
	cudaFree(d_result);
}
	return 0;
}
