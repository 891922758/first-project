#include "cudecompose.h"

int main(int argc,char *argv[]){
	size_t size = sizeof(dt);
	int m = 2;
	int n = 2;
	int k = 2;
	int r = 2;
	int r1 = 2;
	int r2 = 2;
	int r3 = 2;
	dt *X,*A,*B,*C,*core;
	cudaHostAlloc((void**)&X,size*m*n*k,0);
	cudaHostAlloc((void**)&B,size*n*r,0);
	cudaHostAlloc((void**)&C,size*k*r,0);
	cudaHostAlloc((void**)&A,size*m*r,0);
	cudaHostAlloc((void**)&core,size*r1*r2*r3,0);

	srand(2);
	for(int i = 0;i<m*n*k;i++){
		X[i] =float(int(10*(((float) rand())/RAND_MAX - 0.5)));
	}
	for(int i = 0;i<m*r;i++){
		A[i] =float(int(10*(((float) rand())/RAND_MAX - 0.5)));
	}
	for(int i = 0;i<n*r;i++){
		B[i] =float(int(10*(((float) rand())/RAND_MAX - 0.5)));
	}
	for(int i = 0;i<k*r;i++){
		C[i] =float(int(10*(((float) rand())/RAND_MAX - 0.5)));
	}
	for(int i = 0;i<r1*r2*r3;i++){
		core[i] =float(int(10*(((float) rand())/RAND_MAX - 0.5)));
	}

	dt *d_X;
	cudaMalloc((void**)&d_X,size*m*n*k);
	cudaMemcpyAsync(d_X,X,size*m*n*k,cudaMemcpyHostToDevice,0);
	dt *d_X1;
	cudaMalloc((void**)&d_X1,size*m*n*k);
	dt *d_core;
	cudaMalloc((void**)&d_core,size*r1*r2*r3);
	
	dt *d_A,*d_C,*d_B;
	
	cudaMalloc((void**)&d_A,size*m*r);
	cudaMalloc((void**)&d_B,size*n*r);
	cudaMalloc((void**)&d_C,size*k*r);

	cudaMemcpyAsync(d_C,C,size*k*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_B,B,size*n*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_A,A,size*m*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_core,core,size*r1*r2*r3,cudaMemcpyHostToDevice,0);

	cudaDeviceSynchronize();
//	printTensor(d_X,m,n*k,1);
	printTensor(d_A,m,r,1);
	printTensor(d_B,n,r,1);
	printTensor(d_C,k,r,1);
	printTensor(d_core,r1,r2,r3);

//	gencp(d_X1,d_A,d_B,d_C,m,n,k,r);
//	tminust(d_X1,d_X,m*n*k);

	gentucker(d_X1,d_core,d_A,d_B,d_C,m,n,k,r,r,r);
	printTensor(d_X1,m,n,k);
	
	dt error = 0.0;
	rse(d_X,d_X1,m*n*k,&error);
	cout<<error<<endl;

	cudaFreeHost(X);
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFreeHost(A);
	cudaFreeHost(core);
	cudaFree(d_A);
	cudaFree(d_X);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_X1);
	cudaFree(d_core);

}
