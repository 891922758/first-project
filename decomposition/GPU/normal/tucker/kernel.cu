#include "head.h"
__global__  void floattohalf(dt *AA,half *BB,long m){
	long i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<m){
		BB[i]=__float2half(AA[i]);
	}
	__syncthreads();
}
void f2h(dt *A,half *B,long num){
	dim3 threads(512,1,1);
	dim3 blocks((num+512-1)/512,1,1);	
	floattohalf<<<blocks,threads>>>(A,B,num);
}
__global__ void elemin(dt *A,dt *B, long n){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		B[i] = A[i] - B[i];	
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

void printTensor(dt *d_des,long m,long n,long l){
	dt *des = new dt[m*n*l]();
	cudaMemcpy(des,d_des,sizeof(dt)*m*n*l,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for(int k = 0;k<l;k++){
		for(int i = 0;i<n;i++){
			for(int j = 0;j<m;j++){
				cout<<des[k*m*n+i*m+j]<<" ";
			}
			cout<<endl;
		}
		cout<<"~~~~~~~~~~~~~~~~"<<endl;
	}
	delete[] des;des=nullptr;

}

__global__ void elepro(dt *AA,dt *BB,dt *CC,int m){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	const int temp = blockDim.x*gridDim.x;
	while(i<m){
		CC[i] = AA[i]*BB[i];   //rewrite former variable
		i+=temp;
	}
	__syncthreads();
}
__global__ void elepro3(dt *AA,dt *BB,dt *CC,dt *DD,int m){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	const int temp = blockDim.x*gridDim.x;
	while(i<m){
		DD[i] = AA[i]*BB[i]*CC[i];   //rewrite former variable
		i+=temp;
	}
	__syncthreads();
}

__global__ void initIdeMat(dt *AA,int m){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	const int temp = blockDim.x*gridDim.x;
	while(i<m*m){
		int row = i%m;
		int col = i/m;
		if(row==col){
			AA[col*m+row] = 1;
		}else{
			AA[col*m+row] = 0;
		}
		i+=temp;
	}
	__syncthreads();
}

void gencpTensor(dt *T,long a,long b,long c,long r){
	dt *AA,*BB,*CC;		
	cudaHostAlloc((void**)&AA,sizeof(dt)*a*r,0);
	cudaHostAlloc((void**)&BB,sizeof(dt)*b*r,0);
	cudaHostAlloc((void**)&CC,sizeof(dt)*c*r,0);
	for(long i = 0;i<a*r;i++){
		AA[i]=rand()*0.1/(RAND_MAX);
	}
	for(long i = 0;i<b*r;i++){
		BB[i]=rand()*0.1/(RAND_MAX);
	}
	for(long i = 0;i<c*r;i++){
		CC[i]=rand()*0.1/(RAND_MAX);
	}
	dt *d_T,*d_CC,*d_BB,*d_AA;
	cudaMalloc((void**)&d_AA,sizeof(dt)*a*r);
	cudaMalloc((void**)&d_BB,sizeof(dt)*b*r);
	cudaMalloc((void**)&d_CC,sizeof(dt)*c*r);
	cudaMalloc((void**)&d_T,sizeof(dt)*a*b*c);
	cudaMemcpyAsync(d_BB,BB,sizeof(dt)*b*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_CC,CC,sizeof(dt)*c*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_AA,AA,sizeof(dt)*a*r,cudaMemcpyHostToDevice,0);
	dt *d_CKRB;
	cudaMalloc((void**)&d_CKRB,sizeof(dt)*c*r*b);
	cudaDeviceSynchronize();

	//X1=A*(CkrB)'  a*r  r*(bc)
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,c,1,&alpha,d_BB,b,b,d_CC,c,c,&beta,d_CKRB,b,b*c,r);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b*c,r,&alpha,d_AA,a,d_CKRB,b*c,&beta,d_T,a);
	cudaMemcpyAsync(T,d_T,sizeof(dt)*a*b*c,cudaMemcpyDeviceToHost,0);
	cudaDeviceSynchronize();

	cudaFree(d_AA);
	cudaFree(d_BB);
	cudaFree(d_CC);
	cudaFree(d_CKRB);
	cudaFree(d_T);
	cudaFreeHost(AA);
	cudaFreeHost(BB);
	cudaFreeHost(CC);
	cublasDestroy(handle);
}
void gencpTensor4(dt *T,long a,long b,long c,long d,long r){
	dt *AA,*BB,*CC,*DD;		
	cudaHostAlloc((void**)&AA,sizeof(dt)*a*r,0);
	cudaHostAlloc((void**)&BB,sizeof(dt)*b*r,0);
	cudaHostAlloc((void**)&CC,sizeof(dt)*c*r,0);
	cudaHostAlloc((void**)&DD,sizeof(dt)*d*r,0);
	for(int i = 0;i<a*r;i++){
		AA[i]=rand()*0.1/(RAND_MAX);
	}
	for(int i = 0;i<b*r;i++){
		BB[i]=rand()*0.1/(RAND_MAX);
	}
	for(int i = 0;i<c*r;i++){
		CC[i]=rand()*0.1/(RAND_MAX);
	}
	for(int i = 0;i<d*r;i++){
		DD[i]=rand()*0.1/(RAND_MAX);
	}
	dt *d_T,*d_CC,*d_BB,*d_AA,*d_DD;
	cudaMalloc((void**)&d_AA,sizeof(dt)*a*r);
	cudaMalloc((void**)&d_BB,sizeof(dt)*b*r);
	cudaMalloc((void**)&d_CC,sizeof(dt)*c*r);
	cudaMalloc((void**)&d_DD,sizeof(dt)*d*r);
	cudaMalloc((void**)&d_T,sizeof(dt)*a*b*c*d);
	cudaMemcpyAsync(d_BB,BB,sizeof(dt)*b*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_CC,CC,sizeof(dt)*c*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_AA,AA,sizeof(dt)*a*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_DD,DD,sizeof(dt)*d*r,cudaMemcpyHostToDevice,0);
	dt *d_CKRB,*d_kr;
	cudaMalloc((void**)&d_CKRB,sizeof(dt)*c*r*b);
	cudaMalloc((void**)&d_kr,sizeof(dt)*c*r*b*d);
	cudaDeviceSynchronize();

	//X1=A*(DkrCkrB)'  a*r  r*(bcd)
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,c,1,&alpha,d_BB,b,b,d_CC,c,c,&beta,d_CKRB,b,b*c,r);
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,b*c,d,1,&alpha,d_CKRB,b*c,b*c,d_DD,d,d,&beta,d_kr,b*c,b*c*d,r);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b*c*d,r,&alpha,d_AA,a,d_kr,b*c*d,&beta,d_T,a);
	cudaMemcpyAsync(T,d_T,sizeof(dt)*a*b*c*d,cudaMemcpyDeviceToHost,0);
	cudaDeviceSynchronize();

	cudaFree(d_AA);
	cudaFree(d_BB);
	cudaFree(d_CC);
	cudaFree(d_CKRB);
	cudaFree(d_kr);
	cudaFree(d_T);
	cudaFreeHost(AA);
	cudaFreeHost(BB);
	cudaFreeHost(CC);
	cublasDestroy(handle);
}

void gentuTensor(dt *T,long a,long b,long c,long r1,long r2,long r3){
	
	dt *A,*B,*C,*G;		
	cudaHostAlloc((void**)&A,sizeof(dt)*a*r1,0);
	cudaHostAlloc((void**)&B,sizeof(dt)*b*r2,0);
	cudaHostAlloc((void**)&C,sizeof(dt)*c*r3,0);
	cudaHostAlloc((void**)&G,sizeof(dt)*r1*r2*r3,0);
	srand((unsigned)time(NULL));
	for(int i = 0;i<a*r1;i++){
		A[i]=rand()*0.1/(RAND_MAX);
	}
	for(int i = 0;i<b*r2;i++){
		B[i]=rand()*0.1/(RAND_MAX);
	}
	for(int i = 0;i<c*r3;i++){
		C[i]=rand()*0.1/(RAND_MAX);
	}
	for(int i = 0;i<r1*r2*r3;i++){
		G[i]=rand()*0.1/(RAND_MAX);
	}
	dt *d_T,*d_C,*d_B,*d_A,*d_G;
	cudaMalloc((void**)&d_A,sizeof(dt)*a*r1);
	cudaMalloc((void**)&d_B,sizeof(dt)*b*r2);
	cudaMalloc((void**)&d_C,sizeof(dt)*c*r3);
	cudaMalloc((void**)&d_T,sizeof(dt)*a*b*c);
	cudaMalloc((void**)&d_G,sizeof(dt)*r1*r2*r3);
	cudaMemcpyAsync(d_B,B,sizeof(dt)*b*r2,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_C,C,sizeof(dt)*c*r3,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_A,A,sizeof(dt)*a*r1,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_G,G,sizeof(dt)*r1*r2*r3,cudaMemcpyHostToDevice,0);
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);

	dt *d_coreU1,*d_coreU1U2;
	cudaMalloc((void**)&d_coreU1,sizeof(dt)*a*r2*r3);
	cudaMalloc((void**)&d_coreU1U2,sizeof(dt)*a*b*r3);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,a,r2*r3,r1,&alpha,d_A,a,d_G,r1,&beta,d_coreU1,a);
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b,r2,&alpha,d_coreU1,a,a*r2,d_B,b,0,&beta,d_coreU1U2,a,a*b,r3);
	//cout<<"coreU1U2"<<endl; printTensor(d_coreU1U2,a,b,r3);
	//a*b*r3 c*r3 rec a*b*c
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a*b,c,r3,&alpha,d_coreU1U2,a*b,d_C,c,&beta,d_T,a*b);
	cudaMemcpyAsync(T,d_T,sizeof(dt)*a*b*c,cudaMemcpyDeviceToHost,0);
	cudaDeviceSynchronize();

	cudaFree(d_coreU1);
	cudaFree(d_coreU1U2);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_G);
	cudaFree(d_T);

	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFreeHost(G);
	cublasDestroy(handle);
}

void gentuTensor4(dt *T,long a,long b,long c,long d,long r1,long r2,long r3,long r4){
	
	dt *A,*B,*C,*D,*G;		
	cudaHostAlloc((void**)&A,sizeof(dt)*a*r1,0);
	cudaHostAlloc((void**)&B,sizeof(dt)*b*r2,0);
	cudaHostAlloc((void**)&C,sizeof(dt)*c*r3,0);
	cudaHostAlloc((void**)&D,sizeof(dt)*d*r4,0);
	cudaHostAlloc((void**)&G,sizeof(dt)*r1*r2*r3*r4,0);
	srand((unsigned)time(NULL));
	for(int i = 0;i<a*r1;i++){
		A[i]=rand()*0.1/(RAND_MAX);
	}
	for(int i = 0;i<b*r2;i++){
		B[i]=rand()*0.1/(RAND_MAX);
	}
	for(int i = 0;i<c*r3;i++){
		C[i]=rand()*0.1/(RAND_MAX);
	}
	for(int i = 0;i<d*r4;i++){
		D[i]=rand()*0.1/(RAND_MAX);
	}
	for(int i = 0;i<r1*r2*r3;i++){
		G[i]=rand()*0.1/(RAND_MAX);
	}
	dt *d_T,*d_C,*d_B,*d_A,*d_D,*d_G;
	cudaMalloc((void**)&d_A,sizeof(dt)*a*r1);
	cudaMalloc((void**)&d_B,sizeof(dt)*b*r2);
	cudaMalloc((void**)&d_C,sizeof(dt)*c*r3);
	cudaMalloc((void**)&d_D,sizeof(dt)*d*r4);
	cudaMalloc((void**)&d_T,sizeof(dt)*a*b*c*d);
	cudaMalloc((void**)&d_G,sizeof(dt)*r1*r2*r3*r4);
	cudaMemcpyAsync(d_B,B,sizeof(dt)*b*r2,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_C,C,sizeof(dt)*c*r3,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_A,A,sizeof(dt)*a*r1,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_D,D,sizeof(dt)*d*r4,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_G,G,sizeof(dt)*r1*r2*r3*r4,cudaMemcpyHostToDevice,0);
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	dt *d_coreU1,*d_coreU1U2,*d_coreU1U2U3;
	cudaMalloc((void**)&d_coreU1,sizeof(dt)*a*r2*r3*r4);
	cudaMalloc((void**)&d_coreU1U2,sizeof(dt)*a*b*r3*r4);
	cudaMalloc((void**)&d_coreU1U2U3,sizeof(dt)*a*b*c*r4);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,a,r2*r3*r4,r1,&alpha,d_A,a,d_G,r1,&beta,d_coreU1,a);
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b,r2,&alpha,d_coreU1,a,a*r2,d_B,b,0,&beta,d_coreU1U2,a,a*b,r3*r4);
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a*b,c,r3,&alpha,d_coreU1U2,a*b,a*b*r3,d_C,c,0,&beta,d_coreU1U2U3,a*b,a*b*c,r4);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a*b*c,d,r4,&alpha,d_coreU1U2U3,a*b*c,d_D,d,&beta,d_T,a*b*c);
	cudaMemcpyAsync(T,d_T,sizeof(dt)*a*b*c*d,cudaMemcpyDeviceToHost,0);
	cudaDeviceSynchronize();

	cudaFree(d_coreU1);
	cudaFree(d_coreU1U2);
	cudaFree(d_coreU1U2U3);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_D);
	cudaFree(d_G);
	cudaFree(d_T);

	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFreeHost(D);
	cudaFreeHost(G);
	cublasDestroy(handle);
}
