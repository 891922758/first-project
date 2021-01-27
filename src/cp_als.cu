#include "cudecompose.h"

void cp_als(dt *d_X,dt *d_A,dt *d_B,dt *d_C,long a,long b,long c,long r){

// X is a*b*c; A is a*r; B is b*r; C is c*r
// we assume they all store as column
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cusolverDnHandle_t cusolverH = NULL;
	cusolverDnCreate(&cusolverH);
	int *d_info = NULL;
	cudaMalloc((void**)&d_info,sizeof(int));
	int *d_Ipiv = NULL; // PA=LU, P is control weather permute
	cudaMalloc((void**)&d_Ipiv,sizeof(int));
	int lwork=0;
	dt *d_work = NULL;

	curandGenerator_t gen;
	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
	curandGenerateUniform(gen,d_B,b*r);
	curandGenerateUniform(gen,d_C,c*r);

	dt *d_X2;
	cudaMalloc((void**)&d_X2,sizeof(dt)*a*b*c);
	t2m(d_X,d_X2,2,a,b,c);
	cudaDeviceSynchronize();
	
	dt *d_ATA,*d_BTB,*d_CTC;
	cudaMalloc((void**)&d_ATA,sizeof(dt)*r*r);
	cudaMalloc((void**)&d_BTB,sizeof(dt)*r*r);
	cudaMalloc((void**)&d_CTC,sizeof(dt)*r*r);

	dt *d_CkrB;
	cudaMalloc((void**)&d_CkrB,sizeof(dt)*b*c*r); //GPU store C kr B
	dt *d_CkrA;
	cudaMalloc((void**)&d_CkrA,sizeof(dt)*a*c*r); //GPU store C kr A
	dt *d_BkrA;
	cudaMalloc((void**)&d_BkrA,sizeof(dt)*a*b*r); //GPU store B kr A
	const int L = 500;

for(int i = 0;i<L;i++){
	// update A
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,c,1,&alpha,d_B,b,b,d_C,c,c,&beta,d_CkrB,b,b*c,r);
	dt *d_At_r;
	cudaMalloc((void**)&d_At_r,sizeof(dt)*a*r); //GPU store (CkrB)'*X1' as right part 
	dt *d_At_l;
	cudaMalloc((void**)&d_At_l,sizeof(dt)*r*r); //GPU store (CTC.*BTB)' as left part

	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,r,a,b*c,&alpha,d_CkrB,b*c,d_X,a,&beta,d_At_r,r);
	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,r,c,&alpha,d_C,c,d_C,c,&beta,d_CTC,r);
	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,r,b,&alpha,d_B,b,d_B,b,&beta,d_BTB,r);

	hadmard(d_CTC,d_BTB,d_At_l,r,r);

	cusolverDnSgetrf_bufferSize(cusolverH,r,r,d_At_l,r,&lwork);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnSgetrf(cusolverH,r,r,d_At_l,r,d_work,d_Ipiv,d_info);
	cusolverDnSgetrs(cusolverH,CUBLAS_OP_N,r,a,d_At_l,r,d_Ipiv,d_At_r,r,d_info);
	cudaDeviceSynchronize();
	cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,a,r,&alpha,d_At_r,r,&beta,d_A,a,d_A,a);

// update B
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,c,1,&alpha,d_A,a,a,d_C,c,c,&beta,d_CkrA,a,a*c,r);
	dt *d_Bt_r;
	cudaMalloc((void**)&d_Bt_r,sizeof(dt)*b*r); //GPU store (CkrA)'*X2' as right part 
	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,r,b,a*c,&alpha,d_CkrA,a*c,d_X2,b,&beta,d_Bt_r,r);

	dt *d_Bt_l;
	cudaMalloc((void**)&d_Bt_l,sizeof(dt)*r*r); //GPU store (CTC.*ATA)' as left part
	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,r,a,&alpha,d_A,a,d_A,a,&beta,d_ATA,r);
	hadmard(d_CTC,d_ATA,d_Bt_l,r,r);
	cudaDeviceSynchronize();

	cusolverDnSgetrf_bufferSize(cusolverH,r,r,d_Bt_l,r,&lwork);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnSgetrf(cusolverH,r,r,d_Bt_l,r,d_work,d_Ipiv,d_info);
	cusolverDnSgetrs(cusolverH,CUBLAS_OP_N,r,b,d_Bt_l,r,d_Ipiv,d_Bt_r,r,d_info);
	cudaDeviceSynchronize();
	cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,b,r,&alpha,d_Bt_r,r,&beta,d_B,b,d_B,b);

//update C
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b,1,&alpha,d_A,a,a,d_B,b,b,&beta,d_BkrA,a,a*b,r);
	dt *d_Ct_r;
	cudaMalloc((void**)&d_Ct_r,sizeof(dt)*c*r); //GPU store (BkrA)'*X3' as right part 
	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,c,a*b,&alpha,d_BkrA,a*b,d_X,a*b,&beta,d_Ct_r,r);

	dt *d_Ct_l;
	cudaMalloc((void**)&d_Ct_l,sizeof(dt)*r*r); //GPU store (CTC.*ATA)' as left part
	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,r,b,&alpha,d_B,b,d_B,b,&beta,d_BTB,r);
	hadmard(d_BTB,d_ATA,d_Ct_l,r,r);
	cudaDeviceSynchronize();

	cusolverDnSgetrf_bufferSize(cusolverH,r,r,d_Ct_l,r,&lwork);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnSgetrf(cusolverH,r,r,d_Ct_l,r,d_work,d_Ipiv,d_info);
	cusolverDnSgetrs(cusolverH,CUBLAS_OP_N,r,c,d_Ct_l,r,d_Ipiv,d_Ct_r,r,d_info);
	cudaDeviceSynchronize();
	cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,c,r,&alpha,d_Ct_r,r,&beta,d_C,c,d_C,c);

/*	if(i == L-1){
		dt error;
		dt *d_recover;
		cudaMalloc((void**)&d_recover,sizeof(dt)*a*b*c);
		gencp(d_recover,d_A,d_B,d_C,a,b,c,r);
		rse(d_X,d_recover,a*b*c,&error);
		cout<<error<<endl;
		cudaFree(d_recover);
	}
*/
	cudaFree(d_At_r);
	cudaFree(d_At_l);
	cudaFree(d_Bt_r);
	cudaFree(d_Bt_l);
	cudaFree(d_Ct_r);
	cudaFree(d_Ct_l);
}

	cudaDeviceSynchronize();
	cudaFree(d_CkrB);
	cudaFree(d_CkrA);
	cudaFree(d_BkrA);
	cudaFree(d_X2);
	cudaFree(d_ATA);
	cudaFree(d_BTB);
	cudaFree(d_CTC);
	cudaFree(d_Ipiv);
	cudaFree(d_info);
	cudaFree(d_work);
	cusolverDnDestroy(cusolverH);
	cublasDestroy(handle);
}
