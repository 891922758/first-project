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

__global__ void tensorToMode3(dt *T1,dt *T2,int m,int n,int k){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<m*n*k){
		int tube = i/(m*n);
		int row = (i-tube*(m*n))%m;
		int col = (i-tube*(m*n))/m;
		T2[k*(col*m+row)+tube] = T1[tube*m*n+col*m+row];
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
	int r = (int)(a*0.1);
//	int r = 2;
	cout<<a<<endl;
	size_t size=sizeof(dt);
	float time1 = 0.0;
	float time2 = 0.0;
	float time3 = 0.0;
	float temp = 0.0;
	dt *A,*B,*X;
	cudaHostAlloc((void**)&X,size*a*b*c,0);
	cudaHostAlloc((void**)&B,size*b*r,0);
	cudaHostAlloc((void**)&A,size*a*r,0);

	for(int i = 0;i<a*b*c;i++){
		X[i] = (((float) rand())/RAND_MAX - 0.5);
	}
	for(int i = 0;i<b*r;i++){
		B[i] = (((float) rand())/RAND_MAX - 0.5);
	}
	for(int i = 0;i<a*r;i++){
		A[i] = (((float) rand())/RAND_MAX - 0.5);
	}
	
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	GPUTimer timer;
	
	dt *d_X;
	cudaMalloc((void**)&d_X,size*a*b*c);
	cudaMemcpyAsync(d_X,X,size*a*b*c,cudaMemcpyHostToDevice,0);
	dt *d_BkrA,*d_A,*d_B,*d_result;
	cudaMalloc((void**)&d_B,size*b*r);
	cudaMalloc((void**)&d_A,size*a*r);
	cudaMalloc((void**)&d_result,size*c*r);
	cudaMemcpyAsync(d_A,A,size*a*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_B,B,size*b*r,cudaMemcpyHostToDevice,0);
	cudaDeviceSynchronize();
	cudaMalloc((void**)&d_BkrA,size*a*b*r);
	half *h_BkrA,*h_X;
	cudaMalloc((void **)&h_X,sizeof(half)*a*b*c);
	cudaMalloc((void **)&h_BkrA,sizeof(half)*a*b*r);

	dim3 thread(512,1,1);
	dim3 block((a*b*c+512-1)/512,1,1);// for tensor matrix
	dim3 block1((a*b*r+512-1)/512,1,1);//for kr

//	cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);
	cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH);
	int L = 10;

for(int iter = 0;iter<L;++iter){
	//warm up
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b,1,&alpha,d_A,a,a,d_B,b,b,&beta,d_BkrA,a,b*a,r);
	cudaDeviceSynchronize();
	
	dt *d_X3;
	cudaMalloc((void**)&d_X3,size*a*b*c);

	timer.start();
	krpro<<<block1,thread>>>(d_B,d_A,d_BkrA,b,a,r);
	cudaDeviceSynchronize();
//	cout<<"unopt-d_BkrA"<<endl;
//	printTensor(d_BkrA,a*b,r,1);
	tensorToMode3<<<block,thread>>>(d_X,d_X3,a,b,c);
	cudaDeviceSynchronize();
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,c,r,a*b,&alpha,d_X3,c,d_BkrA,a*b,&beta,d_result,c);
	cudaDeviceSynchronize();
	time1 = time1+timer.seconds();
//	cout<<"unopt-d_result"<<endl;
//	printTensor(d_result+2*c,2,3,1);
//	printTensor(d_result,c,r,1);
	cudaFree(d_X3);

	timer.start();
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b,1,&alpha,d_A,a,a,d_B,b,b,&beta,d_BkrA,a,b*a,r);
	cudaDeviceSynchronize();
	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,c,r,a*b,&alpha,d_X,a*b,d_BkrA,a*b,&beta,d_result,c);
	cudaDeviceSynchronize();
	time2 = time2+timer.seconds();
//	cout<<"opt-d_result"<<endl;
//	printTensor(d_result+2*c,2,3,1);
//	printTensor(d_result,c,r,1);

	timer.start();
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b,1,&alpha,d_A,a,a,d_B,b,b,&beta,d_BkrA,a,b*a,r);
	cudaDeviceSynchronize();
	temp = timer.seconds();
	f2h(d_BkrA,h_BkrA,a*b*r);
	f2h(d_X,h_X,a*b*c);
	cudaDeviceSynchronize();
	timer.start();
	cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,c,r,a*b,&alpha,h_X,CUDA_R_16F,a*b,h_BkrA,CUDA_R_16F,a*b,&beta,d_result,CUDA_R_32F,c,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	cudaDeviceSynchronize();
	time3 = time3+temp+timer.seconds();
//	cout<<"opt-d_result"<<endl;
//	printTensor(d_result+2*c,2,3,1);
//	printTensor(d_result,c,r,1);

	if(iter == L-1){
		cout<<"mode3-unop = "<<time1/L<<"ms"<<endl;
		cout<<"mode3-op = "<<time2/L<<"ms"<<endl;
		cout<<"mode3-op-tensorcore = "<<time3/L<<"ms"<<endl;
	}
}

	cublasDestroy(handle);	

	cudaFreeHost(X);
	cudaFreeHost(B);
	cudaFreeHost(A);
	cudaFree(h_X);
	cudaFree(h_BkrA);
	cudaFree(d_X);
	cudaFree(d_B);
	cudaFree(d_A);
	cudaFree(d_BkrA);
	cudaFree(d_result);
}
	return 0;
}
