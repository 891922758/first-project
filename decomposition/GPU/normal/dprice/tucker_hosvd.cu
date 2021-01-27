#include "head.h"

void tucker_hosvd(dt *X,dt *core,dt *U1,dt *U2,dt *U3,int a,int b,int c,int r1,int r2,int r3){
	//X is a*b*c, core is r1*r2*r3, U1 is a*r1,U2 b*r2,U3 is c*r3 
	cudaSetDevice(1);
	dt alpha = 1.0;
	dt alpha1 = -1.0;
	dt beta = 0.0;
	dt sh=0.0;
	dt xia=1.0;
	dim3 threads(512,1,1);
	dim3 block0((a*a+512-1)/512,1,1); //for X2'
	cublasHandle_t handle;
	cublasCreate(&handle);
	cusolverDnHandle_t cusolverH = NULL;
	cusolverDnCreate(&cusolverH);
	dt *d_work = NULL;
	int lwork=0;
	int *devInfo=NULL;
	cudaMalloc((void**)&devInfo,sizeof(int));
	int infogpu=0;

	dt *d_X,*d_X1_X1,*d_X2_X2,*d_X3_X3;
	cudaMalloc((void**)&d_X,sizeof(dt)*a*b*c);
	cudaMalloc((void**)&d_X1_X1,sizeof(dt)*a*a);
	cudaMalloc((void**)&d_X2_X2,sizeof(dt)*b*b);
	cudaMalloc((void**)&d_X3_X3,sizeof(dt)*c*c);
	cudaMemcpyAsync(d_X,X,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice,0);
	cudaDeviceSynchronize();
	//cout<<"X"<<endl; printTensor(d_X,a,b,c);

	dt *d_X2,*d_Idemat;
	cudaMalloc((void**)&d_X2,sizeof(dt)*a*b*c);
	cudaMalloc((void**)&d_Idemat,sizeof(dt)*a*a);
	initIdeMat<<<block0,threads>>>(d_Idemat,a);
	cudaDeviceSynchronize();
	//cout<<"Idemat"<<endl; printTensor(d_Idemat,a,a,1);
	cublasDgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,b,a,a,&alpha,d_X,a,a*b,d_Idemat,a,0,&beta,d_X2,b,a*b,c);
	//cout<<"X2"<<endl; printTensor(d_X2,b,a*c,1);

	//compute X1*X1' a*bc * bc*a
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,a,b*c,&alpha,d_X,a,d_X,a,&beta,d_X1_X1,a);
	//cout<<"X1*X1'"<<endl; printTensor(d_X1_X1,a,a,1);
	//compute X2*X2' b*ac * ac*b
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,b,a*c,&alpha,d_X2,b,d_X2,b,&beta,d_X2_X2,b);
	//cout<<"X2*X2'"<<endl; printTensor(d_X2_X2,b,b,1);
	//compute X3*X3' c*ab * ab*c
	cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,c,c,a*b,&alpha,d_X,a*b,d_X,a*b,&beta,d_X3_X3,c);
	//cout<<"X3*X3'"<<endl; printTensor(d_X3_X3,c,c,1);
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
	cusolverDnDsyevdx_bufferSize(cusolverH,jobz,range,uplo,a,d_X1_X1,a,0.0,1e06,1,a,&meig1,d_W1,&lwork);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnDsyevdx(cusolverH,jobz,range,uplo,a,d_X1_X1,a,0.0,1e06,1,a,&meig1,d_W1,d_work,lwork,devInfo);
	cudaMemcpy(&infogpu,devInfo,sizeof(int),cudaMemcpyDeviceToHost);
	cout<<infogpu<<endl;
	cudaFree(d_work);
	cudaDeviceSynchronize();
	//cout<<"first vectores"<<endl; printTensor(d_X1_X1+(a-r1)*a,a,r1,1);

	// turn X2X2 to eigvectores and view as U2
	cusolverDnDsyevdx_bufferSize(cusolverH,jobz,range,uplo,b,d_X2_X2,b,0.0,1e06,1,b,&meig2,d_W2,&lwork);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnDsyevdx(cusolverH,jobz,range,uplo,b,d_X2_X2,b,0.0,1e06,1,b,&meig2,d_W2,d_work,lwork,devInfo);
	cudaMemcpy(&infogpu,devInfo,sizeof(int),cudaMemcpyDeviceToHost);
	cout<<infogpu<<endl;
	cudaFree(d_work);
	cudaDeviceSynchronize();
	//cout<<"second vectors"<<endl; printTensor(d_X2_X2+(b-r2)*b,b,r2,1);
	
	// turn X3X3 to eigvectores and view as U3
	cusolverDnDsyevdx_bufferSize(cusolverH,jobz,range,uplo,c,d_X3_X3,c,0.0,1e06,1,c,&meig3,d_W3,&lwork);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnDsyevdx(cusolverH,jobz,range,uplo,c,d_X3_X3,c,0.0,1e06,1,c,&meig3,d_W3,d_work,lwork,devInfo);
	cudaMemcpy(&infogpu,devInfo,sizeof(int),cudaMemcpyDeviceToHost);
	cout<<infogpu<<endl;
	cudaDeviceSynchronize();
	//cout<<"third vectores"<<endl; printTensor(d_X3_X3+(c-r3)*c,c,r3,1);
	cudaFree(d_W1);
	cudaFree(d_W2);
	cudaFree(d_W3);
	
	// then compute X x1U1 x2U2 x3U3,we need extra two intenal vals and core to store last result 
	// a*b*c  a*r1  b*r2  c*r3
	// X x1U1' =U1'*X1  X1 can obtain direct store as X
	dt *d_XU1,*d_XU1U2,*d_core;
	cudaMalloc((void**)&d_XU1,sizeof(dt)*r1*b*c);
	cudaMalloc((void**)&d_XU1U2,sizeof(dt)*r1*r2*c);
	cudaMalloc((void**)&d_core,sizeof(dt)*r1*r2*r3);
	//X X1 U1 a*b*c a*r1
	cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r1,b*c,a,&alpha,d_X1_X1+(a-r1)*a,a,d_X,a,&beta,d_XU1,r1);
	//cout<<"XU1"<<endl; printTensor(d_XU1,r1,b*c,1);
	//XU1*U2 r1*b *c  b*r2
	cublasDgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,r1,r2,b,&alpha,d_XU1,r1,r1*b,d_X2_X2+(b-r2)*b,b,0,&beta,d_XU1U2,r1,r1*r2,c);
	//cout<<"XU1U2"<<endl; printTensor(d_XU1U2,r1,r2*c,1);
	//XU1U2*U3'  r1*r2*r3 c*r3
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,r1*r2,r3,c,&alpha,d_XU1U2,r1*r2,d_X3_X3+(c-r3)*c,c,&beta,d_core,r1*r2);
	//cout<<"core"<<endl; printTensor(d_core,1,r1*r2*r3,1);
	cudaDeviceSynchronize();

	//recover by X = core X1U1 X2U2 X3U3
	// r1*r2*r3  a*r1 ,b*r2 ,c*r3
	dt *d_coreU1,*d_coreU1U2,*d_rec;
	cudaMalloc((void**)&d_coreU1,sizeof(dt)*a*r2*r3);
	cudaMalloc((void**)&d_coreU1U2,sizeof(dt)*a*b*r3);
	cudaMalloc((void**)&d_rec,sizeof(dt)*a*b*c);
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,a,r2*r3,r1,&alpha,d_X1_X1+(a-r1)*a,a,d_core,r1,&beta,d_coreU1,a);
	//cout<<"coreU1"<<endl; printTensor(d_coreU1,a,r2,r3);
	//a*r2*r3  b*r2 coreU1U2
	cublasDgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b,r2,&alpha,d_coreU1,a,a*r2,d_X2_X2+(b-r2)*b,b,0,&beta,d_coreU1U2,a,a*b,r3);
	//cout<<"coreU1U2"<<endl; printTensor(d_coreU1U2,a,b,r3);
	//a*b*r3 c*r3 rec a*b*c
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a*b,c,r3,&alpha,d_coreU1U2,a*b,d_X3_X3+(c-r3)*c,c,&beta,d_rec,a*b);
//	cout<<"rec"<<endl; printTensor(d_rec,a,b,c);
	cublasDaxpy(handle,a*b*c,&alpha1,d_X,1,d_rec,1);
//	cout<<"rec"<<endl; printTensor(d_rec,a,b,c);
	cublasDnrm2(handle,a*b*c,d_rec,1,&sh);
	cout<<"sh "<<endl; cout<<sh<<endl;
	cublasDnrm2(handle,a*b*c,d_X,1,&xia);
	cudaDeviceSynchronize();
	cout<<"xi "<<endl; cout<<xia<<endl;
	cout<<"error rate "<<sh/xia<<endl;

	cudaDeviceSynchronize();

	cudaFree(d_coreU1);
	cudaFree(d_coreU1U2);
	cudaFree(d_rec);

	// transfer result to Host
	cudaMemcpyAsync(core,d_core,sizeof(dt)*r1*r2*r3,cudaMemcpyDeviceToHost,0);
	cudaMemcpyAsync(U1,d_X1_X1+(a-r1)*a,sizeof(dt)*a*r1,cudaMemcpyDeviceToHost,0);
	cudaMemcpyAsync(U2,d_X2_X2+(b-r2)*b,sizeof(dt)*b*r2,cudaMemcpyDeviceToHost,0);
	cudaMemcpyAsync(U3,d_X3_X3+(c-r3)*c,sizeof(dt)*c*r3,cudaMemcpyDeviceToHost,0);
//	printTensor(d_X1_X1+(a-r1)*a,a,r1,1);
//	printTensor(d_X2_X2+(b-r2)*b,b,r2,1);
//	printTensor(d_X3_X3+(c-r3)*a,c,r3,1);
//	printTensor(d_core,r1,r2,r3);

	cudaDeviceSynchronize();
	cublasDestroy(handle);
	cusolverDnDestroy(cusolverH);
	cudaFree(d_XU1);
	cudaFree(d_XU1U2);
	cudaFree(d_core);
	cudaFree(d_work);
	cudaFree(devInfo);
	cudaFree(d_X);
	cudaFree(d_X1_X1);
	cudaFree(d_X2_X2);
	cudaFree(d_X3_X3);
	cudaFree(d_X2);
	cudaFree(d_Idemat);
	cudaDeviceReset();
}

