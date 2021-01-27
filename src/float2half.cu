
#include "cudecompose.h"

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

