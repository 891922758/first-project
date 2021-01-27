#include "cudecompose.h"

void cp_als4(dt *d_X,dt *d_A,dt *d_B,dt *d_C, dt *d_D,long a,long b,long c,long d,long r){

// X is a*b*c*d; A is a*r; B is b*r; C is c*r D is d*r
// we assume they all store as column
	dt alpha = 1.0;
	dt beta = 0.0;
	dim3 threads(512,1,1);
	dim3 block0((a*a+512-1)/512,1,1); //for X2'
	dim3 block1((r*r+512-1)/512,1,1); //for elepro
	dim3 block2((a*b*a*b+512-1)/512,1,1); //for X3
	dim3 block3((a*b*c*d+512-1)/512,1,1); //for X3
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
	curandGenerateUniform(gen,d_D,d*r);

	dt *d_X2,*d_Idemat;
	cudaMalloc((void**)&d_X2,sizeof(dt)*a*b*c*d);
	cudaMalloc((void**)&d_Idemat,sizeof(dt)*a*a);
	initIdeMat<<<block0,threads>>>(d_Idemat,a);
	cudaDeviceSynchronize();
	cublasSgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,b,a,a,&alpha,d_X,a,a*b,d_Idemat,a,0,&beta,d_X2,b,a*b,c*d);
	dt *d_X3,*d_Idemat3;
	cudaMalloc((void**)&d_X3,sizeof(dt)*a*b*c*d);
	cudaMalloc((void**)&d_Idemat3,sizeof(dt)*a*b*a*b);
	initIdeMat<<<block2,threads>>>(d_Idemat3,a*b);
	cudaDeviceSynchronize();
	cublasSgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,c,a*b,a*b,&alpha,d_X,a*b,a*b*c,d_Idemat3,a*b,0,&beta,d_X3,c,a*b*c,d);
	cudaFree(d_Idemat3);
	cudaFree(d_Idemat);

	dt *d_ATA,*d_BTB,*d_CTC,*d_DTD;
	cudaMalloc((void**)&d_ATA,sizeof(dt)*r*r);
	cudaMalloc((void**)&d_BTB,sizeof(dt)*r*r);
	cudaMalloc((void**)&d_CTC,sizeof(dt)*r*r);
	cudaMalloc((void**)&d_DTD,sizeof(dt)*r*r);
	dt *d_krA;
	cudaMalloc((void**)&d_krA,sizeof(dt)*b*c*d*r);
	dt *d_krB;
	cudaMalloc((void**)&d_krB,sizeof(dt)*a*c*d*r);
	dt *d_krC;
	cudaMalloc((void**)&d_krC,sizeof(dt)*b*a*d*r);
	dt *d_krD;
	cudaMalloc((void**)&d_krD,sizeof(dt)*b*c*a*r);
	dt *d_DkrC; cudaMalloc((void**)&d_DkrC,sizeof(dt)*d*c*r);
	dt *d_BkrA; cudaMalloc((void**)&d_BkrA,sizeof(dt)*b*a*r);
	const int L = 500;
	
	for(int i = 0;i<L;i++){
		//update A; DkrCkrB
		// get krA b*c*d*r
		cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,c,d,1,&alpha,d_C,c,c,d_D,d,d,&beta,d_DkrC,c,d*c,r);
		cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,c*d,1,&alpha,d_B,b,b,d_DkrC,c*d,c*d,&beta,d_krA,b,b*c*d,r);
		dt *d_At_r;  
		cudaMalloc((void**)&d_At_r,sizeof(dt)*a*r); // (krA)'*X1' as right part 
		dt *d_At_l;
		cudaMalloc((void**)&d_At_l,sizeof(dt)*r*r); // (DTD.*CTC.*BTB)' as left part
		cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,r,a,b*c*d,&alpha,d_krA,b*c*d,d_X,a,&beta,d_At_r,r);
		cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,r,c,&alpha,d_C,c,d_C,c,&beta,d_CTC,r);
		cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,r,b,&alpha,d_B,b,d_B,b,&beta,d_BTB,r);
		cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,r,d,&alpha,d_D,d,d_D,d,&beta,d_DTD,r);
		elepro3<<<block1,threads>>>(d_DTD,d_CTC,d_BTB,d_At_l,r*r);
		// (d_At_l)'A'=d_At_r ,due to d_At_l is symc so we don't tran, d_At_r has implity tran
		// r*r     r*a    r*a store as col
		cusolverDnSgetrf_bufferSize(cusolverH,r,r,d_At_l,r,&lwork);
		cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
		cusolverDnSgetrf(cusolverH,r,r,d_At_l,r,d_work,d_Ipiv,d_info);
//		cudaMemcpy(Ipiv,d_Ipiv,sizeof(int)*r,cudaMemcpyDeviceToHost);	
//		cudaMemcpy(&info,d_info,sizeof(int),cudaMemcpyDeviceToHost);	
//		cout<<"information "<<info<<endl;
		cusolverDnSgetrs(cusolverH,CUBLAS_OP_N,r,a,d_At_l,r,d_Ipiv,d_At_r,r,d_info);
		cudaDeviceSynchronize();
		cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,a,r,&alpha,d_At_r,r,&beta,d_A,a,d_A,a);

		//update B; DkrCkrA ;get krB
		cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,c*d,1,&alpha,d_A,a,a,d_DkrC,c*d,c*d,&beta,d_krB,a,a*c*d,r);
		dt *d_Bt_r;  
		cudaMalloc((void**)&d_Bt_r,sizeof(dt)*b*r); // 
		dt *d_Bt_l;
		cudaMalloc((void**)&d_Bt_l,sizeof(dt)*r*r); // krB'*X2'
		cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,r,b,a*c*d,&alpha,d_krB,a*c*d,d_X2,b,&beta,d_Bt_r,r);
		cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,r,a,&alpha,d_A,a,d_A,a,&beta,d_ATA,r);
		elepro3<<<block1,threads>>>(d_DTD,d_CTC,d_ATA,d_Bt_l,r*r);
		cudaDeviceSynchronize();
		// (d_Bt_l)'B'=d_Bt_r ,due to d_Bt_l is symc so we don't tran, d_Bt_r has been  tran
		// r*r     r*b    r*b  store as col
		cusolverDnSgetrf_bufferSize(cusolverH,r,r,d_Bt_l,r,&lwork);
		cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
		cusolverDnSgetrf(cusolverH,r,r,d_Bt_l,r,d_work,d_Ipiv,d_info);
//		cudaMemcpy(&info,d_info,sizeof(int),cudaMemcpyDeviceToHost);	
//		cout<<"information "<<info<<endl;
		cusolverDnSgetrs(cusolverH,CUBLAS_OP_N,r,b,d_Bt_l,r,d_Ipiv,d_Bt_r,r,d_info);
		cudaDeviceSynchronize();
		// we tanspose B' to B in d_B is b*r
		cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,b,r,&alpha,d_Bt_r,r,&beta,d_B,b,d_B,b);
		//update C  ;DkrBkrA ;get krC                                   
		cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b,1,&alpha,d_A,a,a,d_B,b,b,&beta,d_BkrA,a,a*b,r);
		cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,b*a,d,1,&alpha,d_BkrA,a*b,a*b,d_D,d,d,&beta,d_krC,a*b,a*b*d,r);

		dt *d_Ct_r;
		cudaMalloc((void**)&d_Ct_r,sizeof(dt)*c*r); //GPU store (krC)'*X3' as right part 
		cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,r,c,a*b*d,&alpha,d_krC,a*b*d,d_X3,c,&beta,d_Ct_r,r);
		dt *d_Ct_l;
		cudaMalloc((void**)&d_Ct_l,sizeof(dt)*r*r); //GPU store (CTC.*ATA)' as left part
		cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,r,b,&alpha,d_B,b,d_B,b,&beta,d_BTB,r);
		elepro3<<<block1,threads>>>(d_DTD,d_BTB,d_ATA,d_Ct_l,r*r);
		cudaDeviceSynchronize();
		// (d_Ct_l)'C'=d_Ct_r ,due to d_Ct_l is symc so we don't tran, d_Ct_r has been  tran
		// r*r     r*c    r*c  store as col
		cusolverDnSgetrf_bufferSize(cusolverH,r,r,d_Ct_l,r,&lwork);
		cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
		cusolverDnSgetrf(cusolverH,r,r,d_Ct_l,r,d_work,d_Ipiv,d_info);
//		cudaMemcpy(&info,d_info,sizeof(int),cudaMemcpyDeviceToHost);	
//		cout<<"information "<<info<<endl;
		cusolverDnSgetrs(cusolverH,CUBLAS_OP_N,r,c,d_Ct_l,r,d_Ipiv,d_Ct_r,r,d_info);
		cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,c,r,&alpha,d_Ct_r,r,&beta,d_C,c,d_C,c);    //transpose
		cudaDeviceSynchronize();

		//update D
		cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,b*a,c,1,&alpha,d_BkrA,a*b,a*b,d_C,c,c,&beta,d_krD,a*b,a*b*c,r);
		dt *d_Dt_r;
		cudaMalloc((void**)&d_Dt_r,sizeof(dt)*d*r); //GPU store (krD)'*X4' as right part 
		cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,d,a*b*c,&alpha,d_krD,a*b*c,d_X,a*b*c,&beta,d_Dt_r,r);
		dt *d_Dt_l;
		cudaMalloc((void**)&d_Dt_l,sizeof(dt)*r*r); //GPU store (CTC.*ATA)' as left part
		cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,r,c,&alpha,d_C,c,d_C,c,&beta,d_CTC,r);
		elepro3<<<block1,threads>>>(d_CTC,d_BTB,d_ATA,d_Dt_l,r*r);
		// (d_Dt_l)'D'=d_Dt_r ,due to d_Dt_l is symc so we don't tran, d_Dt_r has been  tran
		// r*r     r*d    r*d  store as col
		cusolverDnSgetrf_bufferSize(cusolverH,r,r,d_Dt_l,r,&lwork);
		cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
		cusolverDnSgetrf(cusolverH,r,r,d_Dt_l,r,d_work,d_Ipiv,d_info);
//		cudaMemcpy(&info,d_info,sizeof(int),cudaMemcpyDeviceToHost);	
//		cout<<"information "<<info<<endl;
		cusolverDnSgetrs(cusolverH,CUBLAS_OP_N,r,d,d_Dt_l,r,d_Ipiv,d_Dt_r,r,d_info);
		cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,d,r,&alpha,d_Dt_r,r,&beta,d_D,d,d_D,d);
		cudaDeviceSynchronize();

	if(i == L-1){
		dt error;
		dt *d_recover;
		cudaMalloc((void**)&d_recover,sizeof(dt)*a*b*c*d);
		gencp4(d_recover,d_A,d_B,d_C,d_D,a,b,c,d,r);
		rse(d_X,d_recover,a*b*c*d,&error);
		cout<<error<<endl;
		cudaFree(d_recover);
	}

		cudaFree(d_At_r);
		cudaFree(d_At_l);
		cudaFree(d_Bt_r);
		cudaFree(d_Bt_l);
		cudaFree(d_Ct_r);
		cudaFree(d_Ct_l);
		cudaFree(d_Dt_r);
		cudaFree(d_Dt_l);
	
	}

	cudaDeviceSynchronize();
	cudaFree(d_krB); cudaFree(d_krA); cudaFree(d_krC); cudaFree(d_krD);
	cudaFree(d_DkrC); cudaFree(d_BkrA);
	cudaFree(d_X2); cudaFree(d_X3);
	cudaFree(d_ATA); cudaFree(d_BTB); cudaFree(d_CTC); cudaFree(d_DTD);
	
	cudaFree(d_Ipiv); cudaFree(d_info); cudaFree(d_work);
	cusolverDnDestroy(cusolverH);
	cublasDestroy(handle);
	cudaDeviceReset();

}
