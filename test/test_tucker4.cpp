#include "cudecompose.h"

int main(int argc,char *argv[]){
	size_t size = sizeof(dt);
	long m = 100;
	long n = m;
	long k = m;
	long l = m;

	long r1 = 10;
	long r2 = r1;
	long r3 = r1;
	long r4 = r1;
	dt *X,*A,*B,*C,*D,*core;
	cudaHostAlloc((void**)&X,size*m*n*k*l,0);
	cudaHostAlloc((void**)&B,size*n*r2,0);
	cudaHostAlloc((void**)&C,size*k*r3,0);
	cudaHostAlloc((void**)&A,size*m*r1,0);
	cudaHostAlloc((void**)&D,size*l*r4,0);
	cudaHostAlloc((void**)&core,size*r1*r2*r3*r4,0);

	for(int i = 0;i<m*r1;i++){
		A[i] =((float)rand()/RAND_MAX);
	}
	for(int i = 0;i<n*r2;i++){
		B[i] =((float)rand()/RAND_MAX);
	}
	for(int i = 0;i<k*r3;i++){
		C[i] =((float)rand()/RAND_MAX);
	}
	for(int i = 0;i<l*r4;i++){
		D[i] =((float)rand()/RAND_MAX);
	}
	for(int i = 0;i<r1*r2*r3*r4;i++){
		core[i] =((float)rand()/RAND_MAX);
	}

	dt *d_X;
	cudaMalloc((void**)&d_X,size*m*n*k*l);
	dt *d_A,*d_C,*d_B,*d_D,*d_core;
	cudaMalloc((void**)&d_A,size*m*r1);
	cudaMalloc((void**)&d_B,size*n*r2);
	cudaMalloc((void**)&d_C,size*k*r3);
	cudaMalloc((void**)&d_D,size*l*r4);
	cudaMalloc((void**)&d_core,size*r1*r2*r3*r4);
	cudaMemcpyAsync(d_C,C,size*k*r3,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_B,B,size*n*r2,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_A,A,size*m*r1,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_D,D,size*l*r4,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_core,core,size*r1*r2*r3*r4,cudaMemcpyHostToDevice,0);

// generate related tensor X
	gentucker4(d_X,d_core,d_A,d_B,d_C,d_D,m,n,k,l,r1,r2,r3,r4);

	dt *d_a,*d_b,*d_c,*d_d,*d_core1;
	cudaMalloc((void**)&d_a,size*m*r1);
	cudaMalloc((void**)&d_b,size*n*r2);
	cudaMalloc((void**)&d_c,size*k*r3);
	cudaMalloc((void**)&d_d,size*l*r4);
	cudaMalloc((void**)&d_core1,size*r1*r2*r3*r4);

	tucker_hosvd4(d_X,d_core1,d_a,d_b,d_c,d_d,m,n,k,l,r1,r2,r3,r4);
	cudaDeviceSynchronize();
	
	cudaFreeHost(X);
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFreeHost(A);
	cudaFreeHost(D);
	cudaFreeHost(core);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_D);
	cudaFree(d_core1);
	cudaFree(d_X);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_D);
	cudaFree(d_core);
}
