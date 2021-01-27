#include "cudecompose.h"

__global__ void elepro3(dt *AA,dt *BB,dt *CC,dt *DD,int m){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<m){
		DD[i] = AA[i]*BB[i]*CC[i];   //rewrite former variable
	}
	__syncthreads();
}

__global__ void initIdeMat(dt *AA,int m){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<m*m){
		int row = i%m;
		int col = i/m;
		if(row==col){
			AA[col*m+row] = 1;
		}else{
			AA[col*m+row] = 0;
		}
	}
	__syncthreads();
}

__global__ void elemin(dt *A,dt *B, long n){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		B[i] = A[i] - B[i];	
	}
    __syncthreads();
}
