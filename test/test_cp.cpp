#include "cudecompose.h"

int main(int argc,char *argv[]){
	size_t size = sizeof(dt);
	long m = 100;
	long n = m;
	long k = m;
	long r = 10;
	dt *X,*A,*B,*C;
	cudaHostAlloc((void**)&X,size*m*n*k,0);
	cudaHostAlloc((void**)&B,size*n*r,0);
	cudaHostAlloc((void**)&C,size*k*r,0);
	cudaHostAlloc((void**)&A,size*m*r,0);

	for(int i = 0;i<m*r;i++){
		A[i] =((float)rand()/RAND_MAX);
	}
	for(int i = 0;i<n*r;i++){
		B[i] =((float)rand()/RAND_MAX);
	}
	for(int i = 0;i<k*r;i++){
		C[i] =((float)rand()/RAND_MAX);
	}

	dt *d_X;
	cudaMalloc((void**)&d_X,size*m*n*k);
	dt *d_A,*d_C,*d_B;
	cudaMalloc((void**)&d_A,size*m*r);
	cudaMalloc((void**)&d_B,size*n*r);
	cudaMalloc((void**)&d_C,size*k*r);
	cudaMemcpyAsync(d_C,C,size*k*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_B,B,size*n*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_A,A,size*m*r,cudaMemcpyHostToDevice,0);

// generate related tensor X
	gencp(d_X,d_A,d_B,d_C,m,n,k,r);

	dt *d_a,*d_b,*d_c;
	cudaMalloc((void**)&d_a,size*m*r);
	cudaMalloc((void**)&d_b,size*n*r);
	cudaMalloc((void**)&d_c,size*k*r);

// cp decomposition for d_a,d_b,d_c
	cp_als(d_X,d_a,d_b,d_c,m,n,k,r);
	cudaDeviceSynchronize();
	
	cudaFreeHost(X);
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFreeHost(A);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_X);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
