
#include "cudecompose.h"

void ttm(dt *d_X, dt *d_U,dt *d_XU, long flag, long a, long b, long c, long d){

	dt alpha = 1.0;
	dt beta = 0.0;

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH);

	half *h_X;
	cudaMalloc((void**)&h_X,sizeof(half)*a*b*c);
	f2h(d_X,h_X,a*b*c);

	if(flag == 1){
		half *h_U;
		cudaMalloc((void**)&h_U,sizeof(half)*d*a);
		f2h(d_U,h_U,d*a);
		cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,d,b*c,a,&alpha,h_U,CUDA_R_16F,d,h_X,CUDA_R_16F,a,&beta,d_XU,CUDA_R_32F,d,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		cudaDeviceSynchronize();
		cudaFree(h_U);

	}else if (flag == 2){
		half *h_U;
		cudaMalloc((void**)&h_U,sizeof(half)*d*b);
		f2h(d_U,h_U,d*b);
		cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,d,b,&alpha,h_X,CUDA_R_16F,a,a*b,h_U,CUDA_R_16F,d,0,&beta,d_XU,CUDA_R_32F,a,a*d,c,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		cudaDeviceSynchronize();
		cudaFree(h_U);

	}else if (flag == 3){
		half *h_U;
		cudaMalloc((void**)&h_U,sizeof(half)*d*c);
		f2h(d_U,h_U,d*c);
		cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,a*b,d,c,&alpha,h_X,CUDA_R_16F,a*b,h_U,CUDA_R_16F,d,&beta,d_XU,CUDA_R_32F,a*b,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		cudaDeviceSynchronize();
		cudaFree(h_U);

	}else{
		cout<<"can not suppror more than 3"<<endl;
	}

	cublasDestroy(handle);
	cudaFree(h_X);
}
