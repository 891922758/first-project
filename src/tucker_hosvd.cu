#include "cudecompose.h"

void tucker_hosvd(dt *d_X,dt *d_core,dt *d_U1,dt *d_U2,dt *d_U3,long a,long b,long c,long r1,long r2,long r3){
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

	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,a,b*c,&alpha,d_X,a,d_X,a,&beta,d_X1_X1,a);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,b,a*c,&alpha,d_X2,b,d_X2,b,&beta,d_X2_X2,b);
	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,c,c,a*b,&alpha,d_X,a*b,d_X,a*b,&beta,d_X3_X3,c);
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
	cudaFree(d_work);
	cudaDeviceSynchronize();
	//cout<<"first vectores"<<endl; printTensor(d_X1_X1+(a-r1)*a,a,r1,1);

	// turn X2X2 to eigvectores and view as U2
	cusolverDnSsyevdx_bufferSize(cusolverH,jobz,range,uplo,b,d_X2_X2,b,0.0,1e06,1,b,&meig2,d_W2,&lwork);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnSsyevdx(cusolverH,jobz,range,uplo,b,d_X2_X2,b,0.0,1e06,1,b,&meig2,d_W2,d_work,lwork,devInfo);
	cudaFree(d_work);
	cudaDeviceSynchronize();
	//cout<<"second vectors"<<endl; printTensor(d_X2_X2+(b-r2)*b,b,r2,1);
	
	// turn X3X3 to eigvectores and view as U3
	cusolverDnSsyevdx_bufferSize(cusolverH,jobz,range,uplo,c,d_X3_X3,c,0.0,1e06,1,c,&meig3,d_W3,&lwork);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnSsyevdx(cusolverH,jobz,range,uplo,c,d_X3_X3,c,0.0,1e06,1,c,&meig3,d_W3,d_work,lwork,devInfo);
	cudaFree(d_work);
	//cout<<"third vectores"<<endl; printTensor(d_X3_X3+(c-r3)*c,c,r3,1);
	cudaFree(d_W1);
	cudaFree(d_W2);
	cudaFree(d_W3);

	cudaMemcpy(d_U1,d_X1_X1+(a-r1)*a,sizeof(dt)*a*r1,cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_U2,d_X2_X2+(b-r2)*b,sizeof(dt)*b*r2,cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_U3,d_X3_X3+(c-r3)*c,sizeof(dt)*c*r3,cudaMemcpyDeviceToDevice);

	// then compute X x1U1 x2U2 x3U3,we need extra two intenal vals and core to store last result 
	// a*b*c  a*r1  b*r2  c*r3
	// X x1U1' =U1'*X1  X1 can obtain direct store as X
	dt *d_XU1,*d_XU1U2;
	cudaMalloc((void**)&d_XU1,sizeof(dt)*r1*b*c);
	cudaMalloc((void**)&d_XU1U2,sizeof(dt)*r1*r2*c);
	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r1,b*c,a,&alpha,d_U1,a,d_X,a,&beta,d_XU1,r1);
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,r1,r2,b,&alpha,d_XU1,r1,r1*b,d_U2,b,0,&beta,d_XU1U2,r1,r1*r2,c);
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,r1*r2,r3,c,&alpha,d_XU1U2,r1*r2,d_U3,c,&beta,d_core,r1*r2);
	cudaDeviceSynchronize();
/*
	//recover by X = core X1U1 X2U2 X3U3
	// r1*r2*r3  a*r1 ,b*r2 ,c*r3
	dt error= 0.0;
	dt *d_rec;
	cudaMalloc((void**)&d_rec,sizeof(dt)*a*b*c);
	gentucker(d_rec,d_core,d_U1,d_U2,d_U3,a,b,c,r1,r2,r3);
	rse(d_X,d_rec,a*b*c,&error);
	cout<<error<<endl;
	cudaFree(d_rec);
	cudaDeviceSynchronize();
*/
	cublasDestroy(handle);
	cusolverDnDestroy(cusolverH);
	cudaFree(d_XU1);
	cudaFree(d_XU1U2);
	cudaFree(devInfo);
	cudaFree(d_X1_X1);
	cudaFree(d_X2_X2);
	cudaFree(d_X3_X3);
	cudaFree(d_X2);
}

