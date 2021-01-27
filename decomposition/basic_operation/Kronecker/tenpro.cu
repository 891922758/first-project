
#include <iostream>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>
 
typedef float dt;
using namespace std;

__global__ void kron(dt *M,dt *N,dt *res,long  m, long n,long k,long l){
	 long i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<m*n*k*l){
		long row = i%(m*k);
		long col = i/(m*k);
		res[col*m*k+row] = M[(row/k)+(col/l)*m]*N[(row%k)+(col%l)*k];
	}
    __syncthreads();
}


__global__ void obtainA(dt *M,dt *res,long  m, long n,long k,long l){
	 long i = blockIdx.x*blockDim.x+threadIdx.x;
	const  long temp = blockDim.x*gridDim.x;
	while(i<m*n*l){
		res[i] = M[m*(i/(m*l))+(i%m)];
		i+=temp;
	}
    __syncthreads();
}

__global__ void obtainB(dt *M,dt *res,long  m, long n,long k,long l){
	 long i = blockIdx.x*blockDim.x+threadIdx.x;
	const  long temp = blockDim.x*gridDim.x;
	while(i<n*k*l){
		res[i] = M[i%(k*l)];
		i+=temp;
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
	cudaSetDevice(0);

	for(int hh = 20;hh<=180;hh=hh+20){
		long a = hh;
		long b = a;
		long c = a;
		long d = a;
		dt *A,*B,*C;
		cout<<a<<" "<<b<<" "<<c<<" "<<d<<endl;
		cudaHostAlloc((void**)&A,sizeof(dt)*a*b,0);
		cudaHostAlloc((void**)&B,sizeof(dt)*c*d,0);
		cudaHostAlloc((void**)&C,sizeof(dt)*a*b*c*d,0);
		for(long i = 0;i<a*b;i++){
			A[i] = rand()*0.1/(RAND_MAX*0.1);		//initial Tensor A
		}
		for(long i = 0;i<c*d;i++){
			B[i] = rand()*0.1/(RAND_MAX*0.1);		//initial Tensor A
		}
		for(long i = 0;i<a*b*c*d;i++){
			C[i] = rand()*0.1/(RAND_MAX*0.1);		//initial Tensor A
		}
//printTensor(A,a,b,1);
//printTensor(B,c,d,1);
		dt alpha = 1.0;
		dt beta = 0.0;
		cublasHandle_t handle;
		cublasCreate(&handle);
		dim3 threads(512,1,1);
		dim3 blocks((a*b*c*d+512-1)/512,1,1);
		dim3 bls(b,1,1);
		dt *d_A;
		dt *d_B;
		dt *d_C;
		cudaMalloc((void **)&d_A,sizeof(dt)*a*b);
		cudaMalloc((void **)&d_B,sizeof(dt)*c*d);
		cudaMalloc((void **)&d_C,sizeof(dt)*a*b*c*d);
		cudaMemcpyAsync(d_A,A,sizeof(dt)*a*b,cudaMemcpyHostToDevice,0);
		cudaMemcpyAsync(d_B,B,sizeof(dt)*c*d,cudaMemcpyHostToDevice,0);
		cudaDeviceSynchronize();
		//warm up
		kron<<<blocks,threads>>>(d_A,d_B,d_C,a,b,c,d);
		kron<<<blocks,threads>>>(d_A,d_B,d_C,a,b,c,d);
	cudaEvent_t start,stop;
	dt elapsedTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	int L = 10;
	for(int j = 0;j<L;j++){
	//	for(int i = 0;i<b;i++){
	//		cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,c,a,1,&alpha,d_B,c,c,d_A+i*a,a,0,&beta,d_C+i*a*c*d,c,c*a,d);
	//}

		kron<<<blocks,threads>>>(d_A,d_B,d_C,a,b,c,d);
}
	cudaEventRecord(stop,0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,start,stop);
	cout<<elapsedTime/L<<endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

		cudaFree(d_A);
		cudaFree(d_B);
		cudaMemcpyAsync(C,d_C,sizeof(dt)*a*b*c*d,cudaMemcpyDeviceToHost,0);
		cudaDeviceSynchronize();
//printTensor(C+100,10,1,1);
		cudaFree(d_C);

		cublasDestroy(handle);	

		cudaFreeHost(A);
		cudaFreeHost(B);
		cudaFreeHost(C);
	
	}
	
	return 0;
}
