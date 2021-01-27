#include "cudecompose.h"
__global__ void hardm(dt *M,dt *N,dt *res,long  m, long n){
	 long i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<m*n){
		res[i] = M[i]*N[i];
	}
    __syncthreads();
}

void hadmard(dt *d_A,dt *d_B,dt *d_C,long a,long b){
		dim3 threads(1024,1,1);
		dim3 blocks((a*b+1024-1)/1024,1,1);
		hardm<<<blocks,threads>>>(d_A,d_B,d_C,a,b);
		cudaDeviceSynchronize();
}

__global__ void hardm3(dt *M,dt *N,dt *K, dt *res,long  m, long n){
	 long i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<m*n){
		res[i] = M[i]*N[i]*K[i];
	}
    __syncthreads();
}

void hadmard3(dt *d_A,dt *d_B,dt *d_C,dt *d_ABC,long a,long b){
		dim3 threads(1024,1,1);
		dim3 blocks((a*b+1024-1)/1024,1,1);
		hardm3<<<blocks,threads>>>(d_A,d_B,d_C,d_ABC,a,b);
		cudaDeviceSynchronize();
}
