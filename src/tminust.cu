#include "cudecompose.h"
__global__ void t_t(dt *M,dt *N,long m){
	long i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<m){
		N[i] = M[i]-N[i];
	}
    __syncthreads();
}

void tminust(dt *d_left,dt *d_right,long m){
		dim3 threads(512,1,1);
		dim3 blocks((m+512-1)/512,1,1);
		t_t<<<blocks,threads>>>(d_left,d_right,m);
		cudaDeviceSynchronize();
}

__global__ void initIdeMat(dt *AA,long m){
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

void initMat(dt *d_A,long m){
	dim3 threads(1024,1,1);
	dim3 blocks((m*m+1024-1)/1024,1,1);
	initIdeMat<<<blocks,threads>>>(d_A,m);
	cudaDeviceSynchronize();
}


