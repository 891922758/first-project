#include "cudecompose.h"

int main(int argc,char *argv[]){
	long m = 4;
	long n = 3;
	long k = 2;;
	size_t size=sizeof(dt);
	dt *A;
	cudaHostAlloc((void**)&A,size*m*n*k,0);
	srand(2);
	for(long i = 0;i<m*n*k;i++){
		A[i] = rand()*0.1/(RAND_MAX);		//initial Tensor A
	}
	
	dt *d_A,*d_A1,*d_A2,*d_A3;
	cudaMalloc((void**)&d_A,size*m*n*k);
	cudaMalloc((void**)&d_A1,size*m*n*k);
	cudaMalloc((void**)&d_A2,size*m*n*k);
	cudaMalloc((void**)&d_A3,size*m*n*k);
	cudaMemcpyAsync(d_A,A,size*m*n*k,cudaMemcpyHostToDevice,0);

	t2m(d_A,d_A1,1,m,n,k);
	cudaDeviceSynchronize();
	
	t2m(d_A,d_A2,2,m,n,k);
	cudaDeviceSynchronize();
	
	t2m(d_A,d_A3,3,m,n,k);
	cudaDeviceSynchronize();
	printTensor(d_A,m,n,k);
	printTensor(d_A1,m,n*k,1);
	printTensor(d_A2,n,m*k,1);
	printTensor(d_A3,k,m*n,1);

	cudaFreeHost(A);
	cudaFree(d_A);
	cudaFree(d_A1);
	cudaFree(d_A2);
	cudaFree(d_A3);

	return 0;
}
