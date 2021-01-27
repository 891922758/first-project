#include "cudecompose.h"

void mttkrp(dt *d_X,dt *d_left,dt *d_right,dt *d_XU,long flag,long m,long n,long k,long r){
	
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH);
	
	if (flag == 1){
		// X m*(nk), left k*r, right n*r, lkrr (nk)*r, XU m*r;
		half *h_X;
		cudaMalloc((void **)&h_X,sizeof(half)*m*n*k);
		f2h(d_X,h_X,m*n*k);
	
		dt *d_lkrr;
		cudaMalloc((void**)&d_lkrr,sizeof(dt)*n*k*r);
		half *h_lkrr;
		cudaMalloc((void **)&h_lkrr,sizeof(half)*n*k*r);
		cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,n,k,1,&alpha,d_right,n,n,d_left,k,k,&beta,d_lkrr,n,k*n,r);
		cudaDeviceSynchronize();
		f2h(d_lkrr,h_lkrr,k*n*r);
		cudaDeviceSynchronize();
		cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,r,n*k,&alpha,h_X,CUDA_R_16F,m,h_lkrr,CUDA_R_16F,n*k,&beta,d_XU,CUDA_R_32F,m,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		cudaFree(h_lkrr);
		cudaFree(d_lkrr);
		cudaFree(h_X);
		cudaDeviceSynchronize();

	}else if (flag == 2){
		// X n*(mk), left k*r, right m*r, lkrr (mk)*r, XU n*r;
		dt *d_X2;
		cudaMalloc((void**)&d_X2,sizeof(dt)*m*n*k);
		dt *d_lkrr;
		cudaMalloc((void**)&d_lkrr,sizeof(dt)*m*k*r);
		half *h_lkrr,*h_X2;
		cudaMalloc((void **)&h_lkrr,sizeof(half)*m*k*r);
		cudaMalloc((void **)&h_X2,sizeof(half)*m*n*k);

		cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,m,k,1,&alpha,d_right,m,m,d_left,k,k,&beta,d_lkrr,m,m*k,r);
		cudaDeviceSynchronize();
		t2m(d_X,d_X2,2,m,n,k);
		cudaDeviceSynchronize();
		f2h(d_lkrr,h_lkrr,k*m*r);
		f2h(d_X2,h_X2,m*n*k);
		cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,r,m*k,&alpha,h_X2,CUDA_R_16F,n,h_lkrr,CUDA_R_16F,m*k,&beta,d_XU,CUDA_R_32F,n,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		cudaFree(h_lkrr);
		cudaFree(d_lkrr);
		cudaFree(d_X2);
		cudaFree(h_X2);
		cudaDeviceSynchronize();
	
	}else if (flag == 3){
		// X k*(mn), left n*r, right m*r, lkrr (mn)*r, XU k*r;
	
		half *h_X;
		cudaMalloc((void **)&h_X,sizeof(half)*m*n*k);
		f2h(d_X,h_X,m*n*k);

		dt *d_lkrr;
		cudaMalloc((void**)&d_lkrr,sizeof(dt)*n*m*r);
		half *h_lkrr;
		cudaMalloc((void **)&h_lkrr,sizeof(half)*n*m*r);
		cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,m,n,1,&alpha,d_right,m,m,d_left,n,n,&beta,d_lkrr,m,m*n,r);
		cudaDeviceSynchronize();
		f2h(d_lkrr,h_lkrr,m*n*r);
		cudaDeviceSynchronize();
		cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,k,r,m*n,&alpha,h_X,CUDA_R_16F,m*n,h_lkrr,CUDA_R_16F,m*n,&beta,d_XU,CUDA_R_32F,k,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		cudaFree(h_X);
		cudaFree(h_lkrr);
		cudaFree(d_lkrr);
		cudaDeviceSynchronize();
	
	}else {
		cout<<"no more than 3"<<endl;
	}

	cublasDestroy(handle);	
}
