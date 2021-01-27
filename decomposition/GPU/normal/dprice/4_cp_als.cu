#include "head.h"

void cp_als4(dt *X,dt *A,dt *B,dt *C, dt *D,int a,int b,int c,int d,int r){

// X is a*b*c*d; A is a*r; B is b*r; C is c*r D is d*r
// we assume they all store as column
	dt alpha = 1.0;
	dt alpha1 = -1.0;
	dt beta = 0.0;
	dt sh=0.0;
	dt xia=1.0;
	dim3 threads(512,1,1);
	dim3 block0((a*a+512-1)/512,1,1); //for X2'
	dim3 block1((r*r+512-1)/512,1,1); //for elepro
	dim3 block2((a*b*a*b+512-1)/512,1,1); //for elepro
	cublasHandle_t handle;
	cublasCreate(&handle);
	cusolverDnHandle_t cusolverH = NULL;
	cusolverDnCreate(&cusolverH);
	int info = 0;
	int *d_info = NULL;
	cudaMalloc((void**)&d_info,sizeof(int));
//	int *Ipiv; 
//	cudaHostAlloc((void**)&Ipiv,sizeof(int)*r,0);
	int *d_Ipiv = NULL; // PA=LU, P is control weather permute
	cudaMalloc((void**)&d_Ipiv,sizeof(int));
	int lwork=0;
	dt *d_work = NULL;

	dt *d_X,*d_C,*d_B,*d_A,*d_D;
	dt *d_rec;
	cudaMalloc((void**)&d_rec,sizeof(dt)*b*a*c*d);
	cudaMalloc((void**)&d_B,sizeof(dt)*b*r);
	cudaMalloc((void**)&d_C,sizeof(dt)*c*r);
	cudaMalloc((void**)&d_X,sizeof(dt)*a*b*c*d);
	cudaMalloc((void**)&d_A,sizeof(dt)*a*r);
	cudaMalloc((void**)&d_D,sizeof(dt)*d*r);

/*	curandGenerator_t gen;
	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
	curandGenerateUniformDouble(gen,d_B,b*r);
	curandGenerateUniformDouble(gen,d_C,c*r);
	curandGenerateUniformDouble(gen,d_D,d*r);
*/
	cudaMemcpyAsync(d_X,X,sizeof(dt)*a*b*c*d,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_B,B,sizeof(dt)*b*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_C,C,sizeof(dt)*c*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_D,D,sizeof(dt)*d*r,cudaMemcpyHostToDevice,0);
	cout<<"X"<<endl; printTensor4(d_X,a,b,c,d);

	dt *d_X2,*d_Idemat;
	cudaMalloc((void**)&d_X2,sizeof(dt)*a*b*c*d);
	cudaMalloc((void**)&d_Idemat,sizeof(dt)*a*a);
	initIdeMat<<<block0,threads>>>(d_Idemat,a);
	cudaDeviceSynchronize();
//	cout<<"Idemat"<<endl; printTensor(d_Idemat,a,a,1);
	cublasDgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,b,a,a,&alpha,d_X,a,a*b,d_Idemat,a,0,&beta,d_X2,b,a*b,c*d);
	cout<<"X2"<<endl; printTensor4(d_X2,b,a*c*d,1,1);
	dt *d_X3,*d_Idemat3;
	cudaMalloc((void**)&d_X3,sizeof(dt)*a*b*c*d);
	cudaMalloc((void**)&d_Idemat3,sizeof(dt)*a*b*a*b);
	initIdeMat<<<block2,threads>>>(d_Idemat3,a*b);
	cudaDeviceSynchronize();
	cublasDgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,c,a*b,a*b,&alpha,d_X,a*b,a*b*c,d_Idemat3,a*b,0,&beta,d_X3,c,a*b*c,d);
	cout<<"X3"<<endl; printTensor4(d_X3,c,a*b*d,1,1);
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
	const int L = 1;
	
	for(int i = 0;i<L;i++){
		//update A; DkrCkrB
		// get krA b*c*d*r
		cublasDgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,c,d,1,&alpha,d_C,c,c,d_D,d,d,&beta,d_DkrC,c,d*c,r);
	cout<<"DkrC"<<endl; printTensor4(d_DkrC,c*d,r,1,1);
		cublasDgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,c*d,1,&alpha,d_B,b,b,d_DkrC,c*d,c*d,&beta,d_krA,b,b*c*d,r);
	cout<<"krA"<<endl; printTensor4(d_krA,c*b*d,r,1,1);
		dt *d_At_r;  
		cudaMalloc((void**)&d_At_r,sizeof(dt)*a*r); // (krA)'*X1' as right part 
		dt *d_At_l;
		cudaMalloc((void**)&d_At_l,sizeof(dt)*r*r); // (DTD.*CTC.*BTB)' as left part
		cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,r,a,b*c*d,&alpha,d_krA,b*c*d,d_X,a,&beta,d_At_r,r);
	cout<<"At_r"<<endl; printTensor4(d_At_r,a,r,1,1);
		cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,r,c,&alpha,d_C,c,d_C,c,&beta,d_CTC,r);
	cout<<"CTC"<<endl; printTensor4(d_CTC,r,r,1,1);
		cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,r,b,&alpha,d_B,b,d_B,b,&beta,d_BTB,r);
	cout<<"BTB"<<endl; printTensor4(d_BTB,r,r,1,1);
		cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,r,d,&alpha,d_D,d,d_D,d,&beta,d_DTD,r);
	cout<<"DTD"<<endl; printTensor4(d_DTD,r,r,1,1);
		elepro3<<<block1,threads>>>(d_DTD,d_CTC,d_BTB,d_At_l,r*r);
	cout<<"At_l"<<endl; printTensor4(d_At_l,r,r,1,1);
		// (d_At_l)'A'=d_At_r ,due to d_At_l is symc so we don't tran, d_At_r has implity tran
		// r*r     r*a    r*a store as col
		cusolverDnDgetrf_bufferSize(cusolverH,r,r,d_At_l,r,&lwork);
		cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
		cusolverDnDgetrf(cusolverH,r,r,d_At_l,r,d_work,d_Ipiv,d_info);
//		cudaMemcpy(Ipiv,d_Ipiv,sizeof(int)*r,cudaMemcpyDeviceToHost);	
		cudaMemcpy(&info,d_info,sizeof(int),cudaMemcpyDeviceToHost);	
		cout<<"information "<<info<<endl;
		cusolverDnDgetrs(cusolverH,CUBLAS_OP_N,r,a,d_At_l,r,d_Ipiv,d_At_r,r,d_info);
		cudaDeviceSynchronize();
		cublasDgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,a,r,&alpha,d_At_r,r,&beta,d_A,a,d_A,a);
	cout<<"A"<<endl; printTensor4(d_A,a,r,1,1);

		//update B; DkrCkrA ;get krB
		cublasDgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,c*d,1,&alpha,d_A,a,a,d_DkrC,c*d,c*d,&beta,d_krB,a,a*c*d,r);
		dt *d_Bt_r;  
		cudaMalloc((void**)&d_Bt_r,sizeof(dt)*b*r); // 
		dt *d_Bt_l;
		cudaMalloc((void**)&d_Bt_l,sizeof(dt)*r*r); // krB'*X2'
		cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,r,b,a*c*d,&alpha,d_krB,a*c*d,d_X2,b,&beta,d_Bt_r,r);
		cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,r,a,&alpha,d_A,a,d_A,a,&beta,d_ATA,r);
		elepro3<<<block1,threads>>>(d_DTD,d_CTC,d_ATA,d_Bt_l,r*r);
		cudaDeviceSynchronize();
		// (d_Bt_l)'B'=d_Bt_r ,due to d_Bt_l is symc so we don't tran, d_Bt_r has been  tran
		// r*r     r*b    r*b  store as col
		cusolverDnDgetrf_bufferSize(cusolverH,r,r,d_Bt_l,r,&lwork);
		cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
		cusolverDnDgetrf(cusolverH,r,r,d_Bt_l,r,d_work,d_Ipiv,d_info);
		cudaMemcpy(&info,d_info,sizeof(int),cudaMemcpyDeviceToHost);	
		cout<<"information "<<info<<endl;
		cusolverDnDgetrs(cusolverH,CUBLAS_OP_N,r,b,d_Bt_l,r,d_Ipiv,d_Bt_r,r,d_info);
		cudaDeviceSynchronize();
		// we tanspose B' to B in d_B is b*r
		cublasDgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,b,r,&alpha,d_Bt_r,r,&beta,d_B,b,d_B,b);
	cout<<"B"<<endl; printTensor4(d_B,b,r,1,1);

		//update C  ;DkrBkrA ;get krC                                   
		cublasDgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b,1,&alpha,d_A,a,a,d_B,b,b,&beta,d_BkrA,a,a*b,r);
		cublasDgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,b*a,d,1,&alpha,d_BkrA,a*b,a*b,d_D,d,d,&beta,d_krC,a*b,a*b*d,r);

		dt *d_Ct_r;
		cudaMalloc((void**)&d_Ct_r,sizeof(dt)*c*r); //GPU store (krC)'*X3' as right part 
		cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,r,c,a*b*d,&alpha,d_krC,a*b*d,d_X3,c,&beta,d_Ct_r,r);
		dt *d_Ct_l;
		cudaMalloc((void**)&d_Ct_l,sizeof(dt)*r*r); //GPU store (CTC.*ATA)' as left part
		cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,r,b,&alpha,d_B,b,d_B,b,&beta,d_BTB,r);
		elepro3<<<block1,threads>>>(d_DTD,d_BTB,d_ATA,d_Ct_l,r*r);
		cudaDeviceSynchronize();
		// (d_Ct_l)'C'=d_Ct_r ,due to d_Ct_l is symc so we don't tran, d_Ct_r has been  tran
		// r*r     r*c    r*c  store as col
		cusolverDnDgetrf_bufferSize(cusolverH,r,r,d_Ct_l,r,&lwork);
		cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
		cusolverDnDgetrf(cusolverH,r,r,d_Ct_l,r,d_work,d_Ipiv,d_info);
		cudaMemcpy(&info,d_info,sizeof(int),cudaMemcpyDeviceToHost);	
		cout<<"information "<<info<<endl;
		cusolverDnDgetrs(cusolverH,CUBLAS_OP_N,r,c,d_Ct_l,r,d_Ipiv,d_Ct_r,r,d_info);
		cublasDgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,c,r,&alpha,d_Ct_r,r,&beta,d_C,c,d_C,c);    //transpose
	cout<<"C"<<endl; printTensor4(d_C,c,r,1,1);
		cudaDeviceSynchronize();

		//update D
		cublasDgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,b*a,c,1,&alpha,d_BkrA,a*b,a*b,d_C,c,c,&beta,d_krD,a*b,a*b*c,r);
		dt *d_Dt_r;
		cudaMalloc((void**)&d_Dt_r,sizeof(dt)*d*r); //GPU store (krD)'*X4' as right part 
		cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,d,a*b*c,&alpha,d_krD,a*b*c,d_X,a*b*c,&beta,d_Dt_r,r);
		dt *d_Dt_l;
		cudaMalloc((void**)&d_Dt_l,sizeof(dt)*r*r); //GPU store (CTC.*ATA)' as left part
		cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r,r,c,&alpha,d_C,c,d_C,c,&beta,d_CTC,r);
		elepro3<<<block1,threads>>>(d_CTC,d_BTB,d_ATA,d_Dt_l,r*r);
	cout<<"Dr"<<endl; printTensor4(d_Dt_r,d,r,1,1);
	cout<<"Dl"<<endl; printTensor4(d_Dt_l,r,r,1,1);
		// (d_Dt_l)'D'=d_Dt_r ,due to d_Dt_l is symc so we don't tran, d_Dt_r has been  tran
		// r*r     r*d    r*d  store as col
		cusolverDnDgetrf_bufferSize(cusolverH,r,r,d_Dt_l,r,&lwork);
		cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
		cusolverDnDgetrf(cusolverH,r,r,d_Dt_l,r,d_work,d_Ipiv,d_info);
		cudaMemcpy(&info,d_info,sizeof(int),cudaMemcpyDeviceToHost);	
		cout<<"information "<<info<<endl;
		cusolverDnDgetrs(cusolverH,CUBLAS_OP_N,r,d,d_Dt_l,r,d_Ipiv,d_Dt_r,r,d_info);
		cublasDgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,d,r,&alpha,d_Dt_r,r,&beta,d_D,d,d_D,d);
	cout<<"D"<<endl; printTensor4(d_D,d,r,1,1);
		cudaDeviceSynchronize();

		//recover X through A B C D; X4 = D*(krD)' => X4'=krD*D';
		cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a*b*c,d,r,&alpha,d_krD,a*b*c,d_D,d,&beta,d_rec,a*b*c);
	cout<<"rec"<<endl; printTensor4(d_rec,b*c*a,d,1,1);
		cublasDaxpy(handle,a*b*c*d,&alpha1,d_X,1,d_rec,1);
		cublasDnrm2(handle,a*b*c*d,d_rec,1,&sh);
		//cout<<"shang "<<endl; cout<<sh<<endl;
		cublasDnrm2(handle,a*b*c*d,d_X,1,&xia);
		cudaDeviceSynchronize();
		//cout<<"xia "<<endl; cout<<xia<<endl;
		cout<<"error rate "<<sh/xia<<endl;
		cudaFree(d_At_r);
		cudaFree(d_At_l);
		cudaFree(d_Bt_r);
		cudaFree(d_Bt_l);
		cudaFree(d_Ct_r);
		cudaFree(d_Ct_l);
		cudaFree(d_Dt_r);
		cudaFree(d_Dt_l);
	
	}
	cudaMemcpyAsync(A,d_A,sizeof(dt)*a*r,cudaMemcpyDeviceToHost,0);
	cudaMemcpyAsync(B,d_B,sizeof(dt)*b*r,cudaMemcpyDeviceToHost,0);
	cudaMemcpyAsync(C,d_C,sizeof(dt)*c*r,cudaMemcpyDeviceToHost,0);
	cudaMemcpyAsync(D,d_D,sizeof(dt)*d*r,cudaMemcpyDeviceToHost,0);

	cudaDeviceSynchronize();
	cudaFree(d_krB); cudaFree(d_krA); cudaFree(d_krC); cudaFree(d_krD);
	cudaFree(d_DkrC); cudaFree(d_BkrA);
	cudaFree(d_B); cudaFree(d_X); cudaFree(d_rec); cudaFree(d_C);
	cudaFree(d_A); cudaFree(d_D);
	cudaFree(d_X2); cudaFree(d_X3);
	cudaFree(d_ATA); cudaFree(d_BTB); cudaFree(d_CTC); cudaFree(d_DTD);
	
	cudaFree(d_Ipiv); cudaFree(d_info); cudaFree(d_work);
	cusolverDnDestroy(cusolverH);
	cublasDestroy(handle);
	cudaDeviceReset();

}
