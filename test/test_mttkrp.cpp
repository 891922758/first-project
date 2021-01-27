#include "cudecompose.h"

int main(int argc,char *argv[]){
	size_t size = sizeof(dt);
	int m = 3;
	int n = 3;
	int k = 3;
	int r = 2;
	dt *X,*A,*B,*C;
	cudaHostAlloc((void**)&X,size*m*n*k,0);
	cudaHostAlloc((void**)&B,size*n*r,0);
	cudaHostAlloc((void**)&C,size*k*r,0);
	cudaHostAlloc((void**)&A,size*m*r,0);

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

	dt *d_X;
	cudaMalloc((void**)&d_X,size*m*n*k);
	cudaMemcpyAsync(d_X,X,size*m*n*k,cudaMemcpyHostToDevice,0);
	
	dt *d_A,*d_C,*d_B,*d_result1,*d_result2,*d_result3;
	
	cudaMalloc((void**)&d_A,size*m*r);
	cudaMalloc((void**)&d_B,size*n*r);
	cudaMalloc((void**)&d_C,size*k*r);

	cudaMalloc((void**)&d_result1,size*m*r);
	cudaMalloc((void**)&d_result2,size*n*r);
	cudaMalloc((void**)&d_result3,size*k*r);

	cudaMemcpyAsync(d_C,C,size*k*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_B,B,size*n*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_A,A,size*m*r,cudaMemcpyHostToDevice,0);
	cudaDeviceSynchronize();

//	mttkrp(d_X,d_C,d_B,d_result1,1,m,n,k,r);
//	mttkrp(d_X,d_C,d_A,d_result2,2,m,n,k,r);
	mttkrp(d_X,d_B,d_A,d_result3,3,m,n,k,r);

	printTensor(d_X,m,n,k);
	printTensor(d_A,m,r,1);
	printTensor(d_B,n,r,1);
//	printTensor(d_C,k,r,1);
//	printTensor(d_result1,m,r,1);
//	printTensor(d_result2,n,r,1);
	printTensor(d_result3,k,r,1);

	cudaFreeHost(X);
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFreeHost(A);
	cudaFree(d_A);
	cudaFree(d_X);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_result1);
	cudaFree(d_result2);
	cudaFree(d_result3);

}
