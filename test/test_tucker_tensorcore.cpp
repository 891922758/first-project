#include "cudecompose.h"

int main(int argc,char *argv[]){
	size_t size = sizeof(dt);
	long m = 160;
	long n = m;
	long k = m;
	long r1 = 16;
	long r2 = 16;
	long r3 = 16;
	dt *X,*A,*B,*C,*core;
	cudaHostAlloc((void**)&X,size*m*n*k,0);
	cudaHostAlloc((void**)&B,size*n*r2,0);
	cudaHostAlloc((void**)&C,size*k*r3,0);
	cudaHostAlloc((void**)&A,size*m*r1,0);
	cudaHostAlloc((void**)&core,size*r1*r2*r3,0);

	for(int i = 0;i<m*r1;i++){
		A[i] =((float)rand()/RAND_MAX);
	}
	for(int i = 0;i<n*r2;i++){
		B[i] =((float)rand()/RAND_MAX);
	}
	for(int i = 0;i<k*r3;i++){
		C[i] =((float)rand()/RAND_MAX);
	}
	for(int i = 0;i<r1*r2*r3;i++){
		core[i] =((float)rand()/RAND_MAX);
	}

	dt *d_X;
	cudaMalloc((void**)&d_X,size*m*n*k);
	dt *d_A,*d_C,*d_B,*d_core;
	cudaMalloc((void**)&d_A,size*m*r1);
	cudaMalloc((void**)&d_B,size*n*r2);
	cudaMalloc((void**)&d_C,size*k*r3);
	cudaMalloc((void**)&d_core,size*r1*r2*r3);
	cudaMemcpyAsync(d_C,C,size*k*r3,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_B,B,size*n*r2,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_A,A,size*m*r1,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_core,core,size*r1*r2*r3,cudaMemcpyHostToDevice,0);

// generate related tensor X
	gentucker(d_X,d_core,d_A,d_B,d_C,m,n,k,r1,r2,r3);

	dt *d_a,*d_b,*d_c,*d_core1;
	cudaMalloc((void**)&d_a,size*m*r1);
	cudaMalloc((void**)&d_b,size*n*r2);
	cudaMalloc((void**)&d_c,size*k*r3);
	cudaMalloc((void**)&d_core1,size*r1*r2*r3);

	tucker_tensorcore(d_X,d_core1,d_a,d_b,d_c,m,n,k,r1,r2,r3);
	cudaDeviceSynchronize();
	
	cudaFreeHost(X);
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFreeHost(A);
	cudaFreeHost(core);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_core1);
	cudaFree(d_X);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_core);
}
