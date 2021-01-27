#include "head.h"

void cp_tensorcore(dt *X,dt *A,dt *B,dt *C,long a,long b,long c,long r){
// X is a*b*c; A is a*r; B is b*r; C is c*r
// we assume they all store as column
	dt alpha = 1.0;
//	dt alpha1 = -1.0;
	dt beta = 0.0;
	dt sh=0.0;
	dt xia=1.0;
	dim3 threads(512,1,1);
	dim3 block1((r*r+512-1)/512,1,1); //for elepro
	dim3 block2((a*b*c+512-1)/512,1,1);
	cublasHandle_t handle;
	cublasCreate(&handle);
//	cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);
	cusolverDnHandle_t cusolverH = NULL;
	cusolverDnCreate(&cusolverH);
//	int info = 0;
	int *d_info = NULL;
	cudaMalloc((void**)&d_info,sizeof(int));
//	int *Ipiv; 
//	cudaHostAlloc((void**)&Ipiv,sizeof(int)*r,0);
	int *d_Ipiv = NULL; // PA=LU, P is control weather permute
	cudaMalloc((void**)&d_Ipiv,sizeof(int));
	int lwork=0;
	dt *d_work = NULL;

	dt *d_X,*d_C,*d_B,*d_A;
	cudaMalloc((void**)&d_B,sizeof(dt)*b*r);
	cudaMalloc((void**)&d_C,sizeof(dt)*c*r);
	cudaMalloc((void**)&d_X,sizeof(dt)*a*b*c);
	cudaMalloc((void**)&d_A,sizeof(dt)*a*r);

	curandGenerator_t gen;
	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
	curandGenerateUniform(gen,d_B,b*r);
	curandGenerateUniform(gen,d_C,c*r);
	//cudaMemcpyAsync(d_B,B,sizeof(dt)*b*r,cudaMemcpyHostToDevice,0);
	//cudaMemcpyAsync(d_C,C,sizeof(dt)*c*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_X,X,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice,0);
//	cout<<"B"<<endl; printTensor(d_B,b,r,1);
//	cout<<"C"<<endl; printTensor(d_C,c,r,1);
//	cout<<"X"<<endl; printTensor(d_X,a,b,c);

	dt *d_X2;
	cudaMalloc((void**)&d_X2,sizeof(dt)*a*b*c);
	tensorToMode2<<<block2,threads>>>(d_X,d_X2,a,b,c);
	cudaDeviceSynchronize();
//	cout<<"X2"<<endl; printTensor(d_X2,b,a,c);

	half *h_X;
	half *h_X2;
	cudaMalloc((void**)&h_X,sizeof(half)*a*b*c);
	cudaMalloc((void**)&h_X2,sizeof(half)*a*b*c);
	f2h(d_X,h_X,a*b*c);
	f2h(d_X2,h_X2,a*b*c);

	
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

	dt *d_At_r;
	cudaMalloc((void**)&d_At_r,sizeof(dt)*a*r); //GPU store (CkrB)'*X1' as right part 
	dt *d_At_l;
	cudaMalloc((void**)&d_At_l,sizeof(dt)*r*r); //GPU store (CTC.*BTB)' as left part
	dt *d_Bt_r;
	cudaMalloc((void**)&d_Bt_r,sizeof(dt)*b*r); //GPU store (CkrA)'*X2' as right part 
	dt *d_Bt_l;
	cudaMalloc((void**)&d_Bt_l,sizeof(dt)*r*r); //GPU store (CTC.*ATA)' as left part
	dt *d_Ct_r;
	cudaMalloc((void**)&d_Ct_r,sizeof(dt)*c*r); //GPU store (BkrA)'*X3' as right part 
	dt *d_Ct_l;
	cudaMalloc((void**)&d_Ct_l,sizeof(dt)*r*r); //GPU store (CTC.*ATA)' as left part
	half *h_CkrB,*h_CkrA,*h_BkrA;
	cudaMalloc((void**)&h_CkrB,sizeof(half)*c*b*r);
	cudaMalloc((void**)&h_CkrA,sizeof(half)*a*c*r);
	cudaMalloc((void**)&h_BkrA,sizeof(half)*a*b*r);

	const int L = 10;
for(int i = 0;i<L;i++){
// update A
//	cout<<"unpdta A"<<endl;
// we compute kr(dot) product
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,c,1,&alpha,d_B,b,b,d_C,c,c,&beta,d_CkrB,b,b*c,r);
//	cout<<"CkrB"<<endl; printTensor(d_CkrB,c*b,r,1);
	f2h(d_CkrB,h_CkrB,b*c*r);
// compute (CkrB)'*X1'
//	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,r,a,b*c,&alpha,d_CkrB,b*c,d_X,a,&beta,d_At_r,r);
	cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_T,r,a,b*c,&alpha,h_CkrB,CUDA_R_16F,b*c,h_X,CUDA_R_16F,a,&beta,d_At_r,CUDA_R_32F,r,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//	cout<<"CkrB'*X1'"<<endl;printTensor(d_At_r,r,a,1);
	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,r,c,&alpha,d_C,c,d_C,c,&beta,d_CTC,r);
	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,r,b,&alpha,d_B,b,d_B,b,&beta,d_BTB,r);
//	cout<<"CTC"<<endl; printTensor(d_CTC,r,r,1);
//	cout<<"BTB"<<endl; printTensor(d_BTB,r,r,1);
	// compute (CTC.*BTB)'  
	elepro<<<block1,threads>>>(d_CTC,d_BTB,d_At_l,r*r);
//	cout<<"CTC.*BTB"<<endl;printTensor(d_At_l,r,r,1);

	//then we solve least squares minimization
	// (d_At_l)'A'=d_At_r ,due to d_At_l is symc so we don't tran, d_At_r has implity tran
	// r*r     r*a    r*a store as col
	cusolverDnSgetrf_bufferSize(cusolverH,r,r,d_At_l,r,&lwork);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnSgetrf(cusolverH,r,r,d_At_l,r,d_work,d_Ipiv,d_info);
//	cudaMemcpy(Ipiv,d_Ipiv,sizeof(int)*r,cudaMemcpyDeviceToHost);	
//	cudaMemcpy(&info,d_info,sizeof(int),cudaMemcpyDeviceToHost);	
//	cout<<"information "<<info<<endl;
	cusolverDnSgetrs(cusolverH,CUBLAS_OP_N,r,a,d_At_l,r,d_Ipiv,d_At_r,r,d_info);
	cudaDeviceSynchronize();
//	printTensor(d_At_r,r,a,1);
//now we obtain A' rewrite d_At_r and store as column
// we tanspose A' to A in d_A is a*r
	cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,a,r,&alpha,d_At_r,r,&beta,d_A,a,d_A,a);
//	cout<<"A"<<endl; printTensor(d_A,a,r,1);

// update B
//	cout<<"then updtate B"<<endl;
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,c,1,&alpha,d_A,a,a,d_C,c,c,&beta,d_CkrA,a,a*c,r);
//	cout<<"C kr A"<<endl;	printTensor(d_CkrA,c*a,r,1);
	f2h(d_CkrA,h_CkrA,a*c*r);
// compute (CkrA)'*X2', we have used batch matrix pro to get X2 
//	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,r,b,a*c,&alpha,d_CkrA,a*c,d_X2,b,&beta,d_Bt_r,r);
	cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_T,r,b,a*c,&alpha,h_CkrA,CUDA_R_16F,a*c,h_X2,CUDA_R_16F,b,&beta,d_Bt_r,CUDA_R_32F,r,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//	cout<<"d_Bt_r"<<endl; printTensor(d_Bt_r,r,b,1);

	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,r,a,&alpha,d_A,a,d_A,a,&beta,d_ATA,r);
//	cout<<"A'A"<<endl;	printTensor(d_ATA,r,r,1);
	elepro<<<block1,threads>>>(d_CTC,d_ATA,d_Bt_l,r*r);
	cudaDeviceSynchronize();
//	cout<<"Bt_l"<<endl; printTensor(d_Bt_l,r,r,1);
	//then we solve least squares minimization
	// (d_Bt_l)'B'=d_Bt_r ,due to d_Bt_l is symc so we don't tran, d_Bt_r has been  tran
	// r*r     r*b    r*b  store as col
	cusolverDnSgetrf_bufferSize(cusolverH,r,r,d_Bt_l,r,&lwork);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnSgetrf(cusolverH,r,r,d_Bt_l,r,d_work,d_Ipiv,d_info);
//	cudaMemcpy(&info,d_info,sizeof(int),cudaMemcpyDeviceToHost);	
//	cout<<"information "<<info<<endl;
	cusolverDnSgetrs(cusolverH,CUBLAS_OP_N,r,b,d_Bt_l,r,d_Ipiv,d_Bt_r,r,d_info);
	cudaDeviceSynchronize();
//	printTensor(d_Bt_r,r,b,1);
//now we obtain B' rewrite d_Bt_r and store as column
// we tanspose B' to B in d_B is b*r
	cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,b,r,&alpha,d_Bt_r,r,&beta,d_B,b,d_B,b);
//	cout<<"B"<<endl; printTensor(d_B,b,r,1);

//update C
//	cout<<"update C"<<endl;
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b,1,&alpha,d_A,a,a,d_B,b,b,&beta,d_BkrA,a,a*b,r);
//	cout<<"B kr A"<<endl;	printTensor(d_BkrA,b*a,r,1);
	f2h(d_BkrA,h_BkrA,a*b*r);
// compute (BkrA)'*X3', as Ct_r 
//	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,c,a*b,&alpha,d_BkrA,a*b,d_X,a*b,&beta,d_Ct_r,r);
	cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,c,b*a,&alpha,h_BkrA,CUDA_R_16F,b*a,h_X,CUDA_R_16F,a*b,&beta,d_Ct_r,CUDA_R_32F,r,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//	cout<<"d_Ct_r"<<endl; printTensor(d_Ct_r,r,c,1);

	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,r,b,&alpha,d_B,b,d_B,b,&beta,d_BTB,r);
//	cout<<"B'B"<<endl;	printTensor(d_BTB,r,r,1);
	elepro<<<block1,threads>>>(d_BTB,d_ATA,d_Ct_l,r*r);
	cudaDeviceSynchronize();
//	cout<<"Ct_l"<<endl; printTensor(d_Ct_l,r,r,1);
	//then we solve least squares minimization
	// (d_Ct_l)'C'=d_Ct_r ,due to d_Ct_l is symc so we don't tran, d_Ct_r has been  tran
	// r*r     r*c    r*c  store as col
	cusolverDnSgetrf_bufferSize(cusolverH,r,r,d_Ct_l,r,&lwork);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnSgetrf(cusolverH,r,r,d_Ct_l,r,d_work,d_Ipiv,d_info);
//	cudaMemcpy(&info,d_info,sizeof(int),cudaMemcpyDeviceToHost);	
//	cout<<"information "<<info<<endl;
	cusolverDnSgetrs(cusolverH,CUBLAS_OP_N,r,c,d_Ct_l,r,d_Ipiv,d_Ct_r,r,d_info);
	cudaDeviceSynchronize();
//	printTensor(d_Ct_r,r,c,1);
//now we obtain C' rewrite d_Ct_r and store as column
// we tanspose C' to C in d_C is c*r
	cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,c,r,&alpha,d_Ct_r,r,&beta,d_C,c,d_C,c);
//	cout<<"C"<<endl; printTensor(d_C,c,r,1);

/*	if(i == L-1){
	// recover to X3' which is same to X
	// X3'= (BkrA)*C' 
	dt *d_rec;
	cudaMalloc((void**)&d_rec,sizeof(dt)*b*a*c);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a*b,c,r,&alpha,d_BkrA,a*b,d_C,c,&beta,d_rec,a*b);
	cudaDeviceSynchronize();
	cout<<"d_X "<<endl; printTensor(d_X+100,4,4,1);
	cout<<"d_C "<<endl; printTensor(d_C+100,4,4,1);
	cout<<"d_BkrA "<<endl; printTensor(d_BkrA+100,4,4,1);
	cout<<"recover_rec "<<endl; printTensor(d_rec+100,4,4,1);
	// rec=-1*X+rec
//	cublasSaxpy(handle,a*b*c,&alpha1,d_X,1,d_rec,1);
	elemin<<<block2,threads>>>(d_X,d_rec,a*b*c);
	cout<<"d_X "<<endl; printTensor(d_X+100,4,1,1);
	cout<<"rec-X "<<endl; printTensor(d_rec+100,4,1,1);
	//error rate = norm(res)/norm(X);
	cublasSnrm2(handle,a*b*c,d_rec,1,&sh);
	cout<<"shang "<<endl; cout<<sh<<endl;
	cublasSnrm2(handle,a*b*c,d_X,1,&xia);
	cudaDeviceSynchronize();
	cout<<"xia "<<endl; cout<<xia<<endl;
	cout<<"error rate "<<sh/xia<<endl;
	cudaFree(d_rec);
	}
*/
}
	cudaFree(d_At_r);
	cudaFree(d_At_l);
	cudaFree(d_Bt_r);
	cudaFree(d_Bt_l);
	cudaFree(d_Ct_r);
	cudaFree(d_Ct_l);

	cudaMemcpyAsync(A,d_A,sizeof(dt)*a*r,cudaMemcpyDeviceToHost,0);
	cudaMemcpyAsync(B,d_B,sizeof(dt)*b*r,cudaMemcpyDeviceToHost,0);
	cudaMemcpyAsync(C,d_C,sizeof(dt)*c*r,cudaMemcpyDeviceToHost,0);

	cudaDeviceSynchronize();

	cudaFree(h_X);cudaFree(h_X2);
	cudaFree(h_CkrB); cudaFree(h_CkrA);cudaFree(h_BkrA);
	
	cudaFree(d_CkrB);
	cudaFree(d_CkrA);
	cudaFree(d_BkrA);

	cudaFree(d_B);
	cudaFree(d_X);
	cudaFree(d_C);
	cudaFree(d_A);
	cudaFree(d_X2);

	cudaFree(d_ATA);
	cudaFree(d_BTB);
	cudaFree(d_CTC);
//	cudaFreeHost(Ipiv);
	cudaFree(d_Ipiv);
	cudaFree(d_info);
	cudaFree(d_work);
	cusolverDnDestroy(cusolverH);
	cublasDestroy(handle);
//	cudaDeviceReset();
}
