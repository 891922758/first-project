#include <iostream>
#include <cuda_runtime.h>
#include "GPUTimer.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>

using namespace std;
typedef float dt;

__global__ void krpro(dt *M,dt *N,dt *res,long long m,long long n,long long r){
	//m*r and n*r to (m*n)*r	
	long long i = blockIdx.x*blockDim.x+threadIdx.x;
	const long long temp = blockDim.x*gridDim.x;
	while(i<m*n*r){
		long long row = i%(m*n);
		long long col = i/(m*n);
		res[col*m*n+row] = M[(row/n)+col*m]*N[(row%n)+col*n];
		i+=temp;
	}
    __syncthreads();
}
__global__ void initIdeMat(dt *AA,int m){
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
__global__ void tensorToMode1(dt *T1,dt *T2,int m,int n,int k ){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<m*n*k){
		int tube = i/(m*n);
		int row = (i-tube*(m*n))%m;
		int col = (i-tube*(m*n))/m;
		T2[tube*m*n+col*m+row] = T1[tube*m*n+col*m+row];
	}
	__syncthreads();
	
}

__global__ void tensorToMode2(dt *T1,dt *T2,int m,int n,int k){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<m*n*k){
		int tube = i/(m*n);
		int row = (i-tube*(m*n))%m;
		int col = (i-tube*(m*n))/m;
		T2[tube*m*n+row*n+col] = T1[tube*m*n+col*m+row];
	}
    __syncthreads();
}

__global__ void tensorToMode3(dt *T1,dt *T2,int m,int n,int k){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<m*n*k){
		int tube = i/(m*n);
		int row = (i-tube*(m*n))%m;
		int col = (i-tube*(m*n))/m;
		T2[k*(col*m+row)+tube] = T1[tube*m*n+col*m+row];
	}
    __syncthreads();
}


void printTensor(dt *A,int a,int b,int c){
	for(int i = 0;i<c;i++){
		for(int j = 0;j<a;j++){
			for(int k =0;k<b;k++){
				cout<<A[i*a*b+k*a+j]<<"  ";
			}
			cout<<endl;
		}
		cout<<"-----------------------------------"<<endl;
	}
	cout<<endl;
}

int main(int argc,char *argv[]){
for(int i = 1100;i<2000;i=i+100){
	setDevice(1);
	int m = i;
	int n = m;
	int k = m;
	int r = (int)(m*0.1);
	cout<<m<<endl;
	size_t size=sizeof(dt);
	float time1 = 0.0;
	float time2 = 0.0;
	float time3 = 0.0;
	float time11 = 0.0;
	float time22 = 0.0;
	float time33 = 0.0;
	dt *A,*B,*C;
	cudaHostAlloc((void**)&A,size*m*n*k,0);
	cudaHostAlloc((void**)&B,size*n*r,0);
	cudaHostAlloc((void**)&C,size*k*r,0);
	srand(2);
	for(int i = 0;i<m*n*k;i++){
		A[i] = rand()*0.1/(RAND_MAX*0.1);		//initial Tensor A
	}
	for(int i = 0;i<n*r;i++){
		B[i] = rand()*0.1/(RAND_MAX*0.1);		//initial Tensor A
	}
	for(int i = 0;i<k*r;i++){
		C[i] = rand()*0.1/(RAND_MAX*0.1);		//initial Tensor A
	}
	
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	GPUTimer timer;
	
	dt *d_A;
	cudaMalloc((void**)&d_A,size*m*n*k);
	cudaMemcpyAsync(d_A,A,size*m*n*k,cudaMemcpyHostToDevice,0);
	dt *d_CkrB,*d_C,*d_B,*d_result;
	cudaMalloc((void**)&d_B,size*n*r);
	cudaMalloc((void**)&d_C,size*k*r);
	cudaMalloc((void**)&d_result,size*m*r);
	cudaMemcpyAsync(d_C,C,size*k*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_B,B,size*n*r,cudaMemcpyHostToDevice,0);
	cudaDeviceSynchronize();
	cudaMalloc((void**)&d_CkrB,size*n*k*r);

	dim3 thread(512,1,1);
	dim3 block((m*n*k+512-1)/512,1,1);
	dim3 block1((m*m+512-1)/512,1,1);

for(int iter = 0;iter<10;++iter){
	//warm up
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,k,n,1,&alpha,d_C,k,k,d_B,n,n,&beta,d_CkrB,k,k*n,r);
	cudaDeviceSynchronize();

	dt *d_A1;
	cudaMalloc((void**)&d_A1,size*m*n*k);

	timer.start();
	tensorToMode1<<<block,thread>>>(d_A,d_A1,m,n,k);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,r,n*k,&alpha,d_A1,m,d_CkrB,n*k,&beta,d_result,m);
	cudaDeviceSynchronize();
	time1 = time1+timer.seconds();
	cudaFree(d_A1);
	
/*	timer.start();
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,r,n*k,&alpha,d_A,m,d_CkrB,n*k,&beta,d_result,m);
	cudaDeviceSynchronize();
	time11 = time11+timer.seconds();
*/

	dt *d_A2;
	cudaMalloc((void**)&d_A2,size*m*n*k);

	timer.start();
	tensorToMode2<<<block,thread>>>(d_A,d_A2,m,n,k);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,r,m*k,&alpha,d_A2,n,d_CkrB,m*k,&beta,d_result,n);
	cudaDeviceSynchronize();
	time2 = time2+timer.seconds();
	
/*	dt *d_Idemat;
	cudaMalloc((void**)&d_Idemat,sizeof(dt)*m*m);

	timer.start();
	initIdeMat<<<block1,thread>>>(d_Idemat,m);
	cublasSgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,m,m,&alpha,d_A,m,m*n,d_Idemat,m,0,&beta,d_A2,n,n*m,k);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,r,m*k,&alpha,d_A2,n,d_CkrB,m*k,&beta,d_result,n);
	cudaDeviceSynchronize();
	time22 = time22+timer.seconds();
	cudaFree(d_Idemat);
*/
	cudaFree(d_A2);
	
	dt *d_A3;
	cudaMalloc((void**)&d_A3,size*m*n*k);

	timer.start();
	tensorToMode3<<<block,thread>>>(d_A,d_A3,m,n,k);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,k,r,n*m,&alpha,d_A3,k,d_CkrB,n*m,&beta,d_result,k);
	cudaDeviceSynchronize();
	time3 = time3+timer.seconds();
	cudaFree(d_A3);

/*	timer.start();
	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,k,r,m*n,&alpha,d_A,m*n,d_CkrB,m*n,&beta,d_result,k);
	cudaDeviceSynchronize();
	time33 = time33+timer.seconds();
*/

	if(iter == 9){
		cout<<"mode1-unop = "<<time1/10<<"ms"<<endl;
		cout<<"mode1-op = "<<time11/10<<"ms"<<endl;
		cout<<"mode2-unop = "<<time2/10<<"ms"<<endl;
		cout<<"mode2-op = "<<time22/10<<"ms"<<endl;
		cout<<"mode3-unop = "<<time3/10<<"ms"<<endl;
		cout<<"mode3-op = "<<time33/10<<"ms"<<endl;
	}
}

	cublasDestroy(handle);	

	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_CkrB);
	cudaFree(d_result);
	cudaDeviceReset();
}
	return 0;
}
