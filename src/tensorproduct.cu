
#include "cudecompose.h"

void tensor_product(dt *d_A,dt *d_B,dt *d_C,long a, long b,long c,long d){
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	for(long i = 0;i<b;i++){
		cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,c,a,1,&alpha,d_B,c,c,d_A+i*a,a,0,&beta,d_C+i*a*c*d,c,c*a,d);
	}

	cublasDestroy(handle);	
}
