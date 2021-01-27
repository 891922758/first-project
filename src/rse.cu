
#include "cudecompose.h"

void rse(dt *d_X,dt *d_X1,long m,dt *error){
	cublasHandle_t handle;
	cublasCreate(&handle);
	dt sh = 0.0;
	dt xia = 1.0;
	tminust(d_X,d_X1,m);
	cublasSnrm2(handle,m,d_X1,1,&sh);
	cublasSnrm2(handle,m,d_X,1,&xia);
	*error = sh/xia;
	cudaDeviceSynchronize();
	cublasDestroy(handle);
}

void gencp(dt *d_rec,dt *d_A,dt *d_B,dt *d_C,long a,long b,long c,long r){
	// rec = A*CkrB'  a*bc = a*r  bc*r
	cublasHandle_t handle;
	cublasCreate(&handle);
	dt alpha = 1.0;
	dt beta = 0.0;
	dt *d_CkrB;
	cudaMalloc((void**)&d_CkrB,sizeof(dt)*b*c*r);
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,c,1,&alpha,d_B,b,b,d_C,c,c,&beta,d_CkrB,b,b*c,r);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b*c,r,&alpha,d_A,a,d_CkrB,b*c,&beta,d_rec,a);
	cudaDeviceSynchronize();
	cudaFree(d_CkrB);
	cublasDestroy(handle);
}

void gentucker(dt *d_rec,dt *d_core, dt *d_A,dt *d_B,dt *d_C,long a,long b,long c,long r1,long r2,long r3){
	// rec = core x1 A x2 B X3 C
	// a*b*c  r1* r2 *r3  a*r1 b*r2 c*r3
	cublasHandle_t handle;
	cublasCreate(&handle);
	dt alpha = 1.0;
	dt beta = 0.0;
	dt *d_coreA,*d_coreAB;
	cudaMalloc((void**)&d_coreA,sizeof(dt)*a*r2*r3);
	cudaMalloc((void**)&d_coreAB,sizeof(dt)*a*b*r3);

	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,a,r2*r3,r1,&alpha,d_A,a,d_core,r1,&beta,d_coreA,a);
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b,r2,&alpha,d_coreA,a,a*r2,d_B,b,0,&beta,d_coreAB,a,a*b,r3);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a*b,c,r3,&alpha,d_coreAB,a*b,d_C,c,&beta,d_rec,a*b);

	cudaDeviceSynchronize();
	cudaFree(d_coreA);
	cudaFree(d_coreAB);
	cublasDestroy(handle);
}

void gencp4(dt *d_T,dt *d_AA,dt *d_BB,dt *d_CC,dt *d_DD,long a,long b,long c,long d,long r){
//X(1) = A*(DkrCkrB)' a*r r*bcd 
	cublasHandle_t handle;
	cublasCreate(&handle);
	dt alpha = 1.0;
	dt beta = 0.0;

	dt *d_CKRB,*d_kr;
	cudaMalloc((void**)&d_CKRB,sizeof(dt)*c*r*b);
	cudaMalloc((void**)&d_kr,sizeof(dt)*c*r*b*d);
	cudaDeviceSynchronize();

	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,c,1,&alpha,d_BB,b,b,d_CC,c,c,&beta,d_CKRB,b,b*c,r);
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,b*c,d,1,&alpha,d_CKRB,b*c,b*c,d_DD,d,d,&beta,d_kr,b*c,b*c*d,r);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b*c*d,r,&alpha,d_AA,a,d_kr,b*c*d,&beta,d_T,a);
	cudaDeviceSynchronize();

	cudaFree(d_CKRB);
	cudaFree(d_kr);
	cublasDestroy(handle);
}

void gentucker4(dt *d_T,dt *d_G, dt *d_A,dt *d_B,dt *d_C,dt *d_D,long a,long b,long c,long d,long r1,long r2,long r3,long r4){
	cublasHandle_t handle;
	cublasCreate(&handle);
	dt alpha = 1.0;
	dt beta = 0.0;
	dt *d_coreU1,*d_coreU1U2,*d_coreU1U2U3;
	cudaMalloc((void**)&d_coreU1,sizeof(dt)*a*r2*r3*r4);
	cudaMalloc((void**)&d_coreU1U2,sizeof(dt)*a*b*r3*r4);
	cudaMalloc((void**)&d_coreU1U2U3,sizeof(dt)*a*b*c*r4);

	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,a,r2*r3*r4,r1,&alpha,d_A,a,d_G,r1,&beta,d_coreU1,a);
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b,r2,&alpha,d_coreU1,a,a*r2,d_B,b,0,&beta,d_coreU1U2,a,a*b,r3*r4);
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a*b,c,r3,&alpha,d_coreU1U2,a*b,a*b*r3,d_C,c,0,&beta,d_coreU1U2U3,a*b,a*b*c,r4);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a*b*c,d,r4,&alpha,d_coreU1U2U3,a*b*c,d_D,d,&beta,d_T,a*b*c);
	cudaDeviceSynchronize();

	cudaFree(d_coreU1);
	cudaFree(d_coreU1U2);
	cudaFree(d_coreU1U2U3);

	cublasDestroy(handle);
}

