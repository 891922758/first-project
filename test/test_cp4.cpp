#include "cudecompose.h"

int main(int argc,char *argv[]){
	size_t size = sizeof(dt);
	long m = 50;
	long n = m;
	long k = m;
	long l = m;
	long r = 5;
	dt *X,*A,*B,*C,*D;
	cudaHostAlloc((void**)&X,size*m*n*k*l,0);
	cudaHostAlloc((void**)&B,size*n*r,0);
	cudaHostAlloc((void**)&C,size*k*r,0);
	cudaHostAlloc((void**)&A,size*m*r,0);
	cudaHostAlloc((void**)&D,size*l*r,0);

	for(int i = 0;i<m*r;i++){
		A[i] =((float)rand()/RAND_MAX);
	}
	for(int i = 0;i<n*r;i++){
		B[i] =((float)rand()/RAND_MAX);
	}
	for(int i = 0;i<k*r;i++){
		C[i] =((float)rand()/RAND_MAX);
	}
	for(int i = 0;i<l*r;i++){
		D[i] =((float)rand()/RAND_MAX);
	}

	dt *d_X;
	cudaMalloc((void**)&d_X,size*m*n*k*l);
	dt *d_A,*d_C,*d_B,*d_D;
	cudaMalloc((void**)&d_A,size*m*r);
	cudaMalloc((void**)&d_B,size*n*r);
	cudaMalloc((void**)&d_C,size*k*r);
	cudaMalloc((void**)&d_D,size*l*r);
	cudaMemcpyAsync(d_C,C,size*k*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_B,B,size*n*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_A,A,size*m*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_D,D,size*l*r,cudaMemcpyHostToDevice,0);

// generate related tensor X
	gencp4(d_X,d_A,d_B,d_C,d_D,m,n,k,l,r);

	dt *d_a,*d_b,*d_c,*d_d;
	cudaMalloc((void**)&d_a,size*m*r);
	cudaMalloc((void**)&d_b,size*n*r);
	cudaMalloc((void**)&d_c,size*k*r);
	cudaMalloc((void**)&d_d,size*l*r);

// cp decomposition for d_a,d_b,d_c
	cp_als4(d_X,d_a,d_b,d_c,d_d,m,n,k,l,r);
	cudaDeviceSynchronize();
	
	cudaFreeHost(X);
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFreeHost(A);
	cudaFreeHost(D);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_d);
	cudaFree(d_X);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_D);
}
