#include <iostream>
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

__global__ void tensorToMode2(dt *T1,dt *T2,int m,int n,int k){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<m*n*k){
		int tube = i/(m*n);
		int row = (i-tube*(m*n))%m;
		int col = (i-tube*(m*n))/m;
		T2[tube*m*n+row*n+col] = T1[tube*m*n+col*m+row];
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
for(int i = 160;i<1600;i=i+160){
	int a = i;
	int b = a;
	int c = a;
//	int r = 2;
	int r = (int)(a*0.1);
	cout<<a<<endl;
	size_t size=sizeof(dt);
	float time1 = 0.0;
	float time2 = 0.0;
	float time3 = 0.0;
	float temp = 0.0;
	dt *A,*X,*C;
	cudaHostAlloc((void**)&X,size*a*b*c,0);
	cudaHostAlloc((void**)&A,size*a*r,0);
	cudaHostAlloc((void**)&C,size*c*r,0);

	for(int i = 0;i<a*b*c;i++){
		X[i] = (((float) rand())/RAND_MAX - 0.5);
	}
	for(int i = 0;i<a*r;i++){
		A[i] = (((float) rand())/RAND_MAX - 0.5);
	}
	for(int i = 0;i<c*r;i++){
		C[i] = (((float) rand())/RAND_MAX - 0.5);
	}
	
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	GPUTimer timer;
	
	dt *d_X;
	cudaMalloc((void**)&d_X,size*a*b*c);
	cudaMemcpyAsync(d_X,X,size*a*b*c,cudaMemcpyHostToDevice,0);
	dt *d_CkrA,*d_C,*d_A,*d_result;
	cudaMalloc((void**)&d_A,size*a*r);
	cudaMalloc((void**)&d_C,size*c*r);
	cudaMalloc((void**)&d_result,size*b*r);
	cudaMemcpyAsync(d_C,C,size*c*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_A,A,size*a*r,cudaMemcpyHostToDevice,0);
	cudaDeviceSynchronize();
	cudaMalloc((void**)&d_CkrA,size*c*a*r);
	half *h_CkrA,*h_X2;
	cudaMalloc((void **)&h_X2,sizeof(half)*a*b*c);
	cudaMalloc((void **)&h_CkrA,sizeof(half)*a*c*r);

	dim3 thread(512,1,1);
	dim3 block((a*b*c+512-1)/512,1,1);// for tensor matrix
	dim3 block1((a*c*r+512-1)/512,1,1);//for kr

//	cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);
	cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH);
	int L = 10;

for(int iter = 0;iter<L;++iter){
	//warm up
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,c,1,&alpha,d_A,a,a,d_C,c,c,&beta,d_CkrA,a,a*c,r);
	cudaDeviceSynchronize();

	dt *d_X2;
	cudaMalloc((void**)&d_X2,size*a*b*c);

	timer.start();
	krpro<<<block1,thread>>>(d_C,d_A,d_CkrA,c,a,r);
	cudaDeviceSynchronize();
//	cout<<"unopt-d_CkrA"<<endl;
//	printTensor(d_CkrA,a*c,r,1);
	tensorToMode2<<<block,thread>>>(d_X,d_X2,a,b,c);
	cudaDeviceSynchronize();
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,b,r,a*c,&alpha,d_X2,b,d_CkrA,a*c,&beta,d_result,b);
	cudaDeviceSynchronize();
	time1 = time1+timer.seconds();
//	cout<<"unopt-d_result"<<endl;
//	printTensor(d_result+2*b,2,3,1);
//	printTensor(d_result,b,r,1);
	
	timer.start();
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,c,1,&alpha,d_A,a,a,d_C,c,c,&beta,d_CkrA,a,a*c,r);
	cudaDeviceSynchronize();
	tensorToMode2<<<block,thread>>>(d_X,d_X2,a,b,c);
	cudaDeviceSynchronize();
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,b,r,a*c,&alpha,d_X2,b,d_CkrA,a*c,&beta,d_result,b);
	cudaDeviceSynchronize();
	time2 =time2+timer.seconds();
//	cout<<"opt-d_result"<<endl;
//	printTensor(d_result+2*b,2,3,1);
//	printTensor(d_result,b,r,1);

	timer.start();
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,c,1,&alpha,d_A,a,a,d_C,c,c,&beta,d_CkrA,a,a*c,r);
	cudaDeviceSynchronize();
	tensorToMode2<<<block,thread>>>(d_X,d_X2,a,b,c);
	cudaDeviceSynchronize();
	temp = timer.seconds();
	f2h(d_CkrA,h_CkrA,c*a*r);
	f2h(d_X2,h_X2,a*b*c);
	timer.start();
	cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,b,r,a*c,&alpha,h_X2,CUDA_R_16F,b,h_CkrA,CUDA_R_16F,a*c,&beta,d_result,CUDA_R_32F,b,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,b,r,a*c,&alpha,d_X2,b,d_CkrA,a*c,&beta,d_result,b);
	cudaDeviceSynchronize();
	time3 =time3+ temp+timer.seconds();
//	cout<<"opt-d_result"<<endl;
//	printTensor(d_result+2*b,2,3,1);
//	printTensor(d_result,b,r,1);

	cudaFree(d_X2);
	
	if(iter == L-1){
		cout<<"mode2-unop = "<<time1/L<<"ms"<<endl;
		cout<<"mode2-op = "<<time2/L<<"ms"<<endl;
		cout<<"mode2-op-tensorcore = "<<time3/L<<"ms"<<endl;
	}
}

	cublasDestroy(handle);	

	cudaFreeHost(X);
	cudaFreeHost(A);
	cudaFreeHost(C);
	cudaFree(h_X2);
	cudaFree(h_CkrA);
	cudaFree(d_X);
	cudaFree(d_A);
	cudaFree(d_C);
	cudaFree(d_CkrA);
	cudaFree(d_result);
}
	return 0;
}
