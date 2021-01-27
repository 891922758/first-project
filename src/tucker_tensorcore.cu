#include "cudecompose.h"

void tucker_tensorcore(dt *d_X,dt *d_core,dt *d_U1,dt *d_U2,dt *d_U3,long a,long b,long c,long r1,long r2,long r3){
	//X is a*b*c, core is r1*r2*r3, U1 is a*r1,U2 b*r2,U3 is c*r3 
	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cusolverDnHandle_t cusolverH = NULL;
	cusolverDnCreate(&cusolverH);
	dt *d_work = NULL;
	int lwork=0;
	int *devInfo=NULL;
	cudaMalloc((void**)&devInfo,sizeof(int));

	dt *d_X1_X1,*d_X2_X2,*d_X3_X3;
	cudaMalloc((void**)&d_X1_X1,sizeof(dt)*a*a);
	cudaMalloc((void**)&d_X2_X2,sizeof(dt)*b*b);
	cudaMalloc((void**)&d_X3_X3,sizeof(dt)*c*c);
	cudaDeviceSynchronize();

	dt *d_X2;
	cudaMalloc((void**)&d_X2,sizeof(dt)*a*b*c);
	t2m(d_X,d_X2,2,a,b,c);
	cudaDeviceSynchronize();
	half *h_X,*h_X2;
	cudaMalloc((void**)&h_X2,sizeof(half)*a*b*c);
	cudaMalloc((void**)&h_X,sizeof(half)*a*b*c);
	f2h(d_X,h_X,a*b*c);
	f2h(d_X2,h_X2,a*b*c);
	cudaFree(d_X2);
	cudaDeviceSynchronize();

	//compute X1*X1' a*bc * bc*a
	cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,a,b*c,&alpha,h_X,CUDA_R_16F,a,h_X,CUDA_R_16F,a,&beta,d_X1_X1,CUDA_R_32F,a,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,a,b*c,&alpha,d_X,a,d_X,a,&beta,d_X1_X1,a);
//	cout<<"X1*X1'"<<endl; printTensor(d_X1_X1,2,3,1);
	//compute X2*X2' b*ac * ac*b
//	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,b,a*c,&alpha,d_X2,b,d_X2,b,&beta,d_X2_X2,b);
	cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,b,a*c,&alpha,h_X2,CUDA_R_16F,b,h_X2,CUDA_R_16F,b,&beta,d_X2_X2,CUDA_R_32F,b,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//	cout<<"X2*X2'"<<endl; printTensor(d_X2_X2,2,3,1);
	//compute X3*X3' c*ab * ab*c
//	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,c,c,a*b,&alpha,d_X,a*b,d_X,a*b,&beta,d_X3_X3,c);
	cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,c,c,b*a,&alpha,h_X,CUDA_R_16F,b*a,h_X,CUDA_R_16F,a*b,&beta,d_X3_X3,CUDA_R_32F,c,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//	cout<<"X3*X3'"<<endl; printTensor(d_X3_X3,2,3,1);
	cudaDeviceSynchronize();
	
	// syevd for U1,U2,U3
	//data prepare for store eigvalue and eigvectors,we only fetch r1 r2 and r3 eigvectors from origin
	dt *d_W1,*d_W2,*d_W3; 
	cudaMalloc((void**)&d_W1,sizeof(dt)*a);
	cudaMalloc((void**)&d_W2,sizeof(dt)*b);
	cudaMalloc((void**)&d_W3,sizeof(dt)*c);
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	cusolverEigRange_t range = CUSOLVER_EIG_RANGE_ALL;
	int meig1=a; int meig2=b; int meig3=c;
	// turn X1X1 to eigvectores and view as U1
	cusolverDnSsyevdx_bufferSize(cusolverH,jobz,range,uplo,a,d_X1_X1,a,0.0,1e06,1,a,&meig1,d_W1,&lwork);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnSsyevdx(cusolverH,jobz,range,uplo,a,d_X1_X1,a,0.0,1e06,1,a,&meig1,d_W1,d_work,lwork,devInfo);
//cudaMemcpy(&infogpu,devInfo,sizeof(int),cudaMemcpyDeviceToHost);
//cout<<infogpu<<endl;
	cudaFree(d_work);
	cudaDeviceSynchronize();
//cout<<"first vectores"<<endl; printTensor(d_X1_X1+(a-r1)*a,2,3,1);

	// turn X2X2 to eigvectores and view as U2
	cusolverDnSsyevdx_bufferSize(cusolverH,jobz,range,uplo,b,d_X2_X2,b,0.0,1e06,1,b,&meig2,d_W2,&lwork);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnSsyevdx(cusolverH,jobz,range,uplo,b,d_X2_X2,b,0.0,1e06,1,b,&meig2,d_W2,d_work,lwork,devInfo);
	cudaFree(d_work);
	cudaDeviceSynchronize();
	
	// turn X3X3 to eigvectores and view as U3
	cusolverDnSsyevdx_bufferSize(cusolverH,jobz,range,uplo,c,d_X3_X3,c,0.0,1e06,1,c,&meig3,d_W3,&lwork);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnSsyevdx(cusolverH,jobz,range,uplo,c,d_X3_X3,c,0.0,1e06,1,c,&meig3,d_W3,d_work,lwork,devInfo);
	cudaFree(d_work);
	cudaFree(d_W1);
	cudaFree(d_W2);
	cudaFree(d_W3);
	
	half *h_U1,*h_U2,*h_U3;
	cudaMalloc((void**)&h_U1,sizeof(half)*a*r1);
	cudaMalloc((void**)&h_U2,sizeof(half)*b*r2);
	cudaMalloc((void**)&h_U3,sizeof(half)*c*r3);
	f2h(d_X1_X1+(a-r1)*a,h_U1,a*r1);
	f2h(d_X2_X2+(b-r2)*b,h_U2,b*r2);
	f2h(d_X3_X3+(c-r3)*c,h_U3,c*r3);
	cudaDeviceSynchronize();
	// then compute X x1U1 x2U2 x3U3,we need extra two intenal vals and core to store last result 
	// a*b*c  a*r1  b*r2  c*r3
	// X x1U1' =U1'*X1  X1 can obtain direct store as X
	half *h_XU1,*h_XU1U2;
	cudaMalloc((void**)&h_XU1,sizeof(half)*r1*b*c);
	cudaMalloc((void**)&h_XU1U2,sizeof(half)*r1*r2*c);
	//X X1 U1 a*b*c a*r1
//	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r1,b*c,a,&alpha,d_X1_X1+(a-r1)*a,a,d_X,a,&beta,d_XU1,r1);
	cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,r1,b*c,a,&alpha,h_U1,CUDA_R_16F,a,h_X,CUDA_R_16F,a,&beta,h_XU1,CUDA_R_16F,r1,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//	cout<<"XU1"<<endl; printTensor(d_XU1,2,3,1);
	//XU1*U2 r1*b *c  b*r2
//	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,r1,r2,b,&alpha,d_XU1,r1,r1*b,d_X2_X2+(b-r2)*b,b,0,&beta,d_XU1U2,r1,r1*r2,c);
	cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,r1,r2,b,&alpha,h_XU1,CUDA_R_16F,r1,r1*b,h_U2,CUDA_R_16F,b,0,&beta,h_XU1U2,CUDA_R_16F,r1,r1*r2,c,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	//cout<<"XU1U2"<<endl; printTensor(d_XU1U2,r1,r2*c,1);
	//XU1U2*U3'  r1*r2*r3 c*r3
//	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,r1*r2,r3,c,&alpha,d_XU1U2,r1*r2,d_X3_X3+(c-r3)*c,c,&beta,d_core,r1*r2);
	cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,r1*r2,r3,c,&alpha,h_XU1U2,CUDA_R_16F,r1*r2,h_U3,CUDA_R_16F,c,&beta,d_core,CUDA_R_32F,r1*r2,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//	cout<<"core"<<endl; printTensor(d_core,1,r1*r2*r3,1);
//	cout<<"core"<<endl; printTensor(d_core,1,5,1);
	cudaDeviceSynchronize();


	//recover by X = core X1U1 X2U2 X3U3
	// r1*r2*r3  a*r1 ,b*r2 ,c*r3
	dt error= 0.0;
	dt *d_rec;
	cudaMalloc((void**)&d_rec,sizeof(dt)*a*b*c);
	gentucker(d_rec,d_core,d_X1_X1+(a-r1)*a,d_X2_X2+(b-r2)*b,d_X3_X3+(c-r3)*c,a,b,c,r1,r2,r3);
	rse(d_X,d_rec,a*b*c,&error);
	cout<<error<<endl;
	cudaFree(d_rec);
	cudaDeviceSynchronize();

	cublasDestroy(handle);
	cusolverDnDestroy(cusolverH);
	cudaFree(h_U1);
	cudaFree(h_U2);
	cudaFree(h_U3);

	cudaFree(h_XU1);
	cudaFree(h_XU1U2);
	cudaFree(devInfo);
	cudaFree(d_X1_X1);
	cudaFree(d_X2_X2);
	cudaFree(d_X3_X3);
	cudaFree(h_X);
	cudaFree(h_X2);
}

