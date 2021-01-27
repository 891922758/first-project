#include "head.h"

void tucker_hooi(dt *X,dt *core,dt *U1,dt *U2,dt *U3,int a,int b,int c,int r1,int r2,int r3){
	//X is a*b*c, core is r1*r2*r3, U1 is a*r1,U2 b*r2,U3 is c*r3 
	dt alpha = 1.0;
	dt alpha1 = -1.0;
	dt beta = 0.0;
	dt sh=0.0;
	dt xia=1.0;
	dim3 threads(512,1,1);
	dim3 block0((a*a+512-1)/512,1,1); //for X2'
	dim3 block1((r1*r1+512-1)/512,1,1);
	cublasHandle_t handle;
	cublasCreate(&handle);
	cusolverDnHandle_t cusolverH = NULL;
	cusolverDnCreate(&cusolverH);
	dt *d_work = NULL;
	int lwork=0;
	int *devInfo=NULL;
	cudaMalloc((void**)&devInfo,sizeof(int));
	int infogpu=0;

	dt *d_X,*d_U1,*d_U2,*d_U3,*d_X2_X2,*d_X3_X3;
	cudaMalloc((void**)&d_X,sizeof(dt)*a*b*c);
	cudaMalloc((void**)&d_U1,sizeof(dt)*a*r1);
	cudaMalloc((void**)&d_U2,sizeof(dt)*b*r2);
	cudaMalloc((void**)&d_U3,sizeof(dt)*c*r3);
	cudaMalloc((void**)&d_X2_X2,sizeof(dt)*b*b);
	cudaMalloc((void**)&d_X3_X3,sizeof(dt)*c*c);
	cudaMemcpyAsync(d_X,X,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice,0);
	cudaDeviceSynchronize();
	//cout<<"X"<<endl; printTensor(d_X,a,b,c);

	dt *d_X2,*d_Idemat,*d_R1den;
	cudaMalloc((void**)&d_X2,sizeof(dt)*a*b*c);
	cudaMalloc((void**)&d_Idemat,sizeof(dt)*a*a);
	cudaMalloc((void**)&d_R1den,sizeof(dt)*r1*r1);
	initIdeMat<<<block0,threads>>>(d_Idemat,a);
	initIdeMat<<<block1,threads>>>(d_R1den,r1);
	cudaDeviceSynchronize();
	//cout<<"Idemat"<<endl; printTensor(d_Idemat,a,a,1);
	cublasDgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,b,a,a,&alpha,d_X,a,a*b,d_Idemat,a,0,&beta,d_X2,b,a*b,c);
	//cout<<"X2"<<endl; printTensor(d_X2,b,a*c,1);

	//compute X2*X2' b*ac * ac*b
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,b,a*c,&alpha,d_X2,b,d_X2,b,&beta,d_X2_X2,b);
	//cout<<"X2*X2'"<<endl; printTensor(d_X2_X2,b,b,1);
	//compute X3*X3' c*ab * ab*c
	cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,c,c,a*b,&alpha,d_X,a*b,d_X,a*b,&beta,d_X3_X3,c);
	//cout<<"X3*X3'"<<endl; printTensor(d_X3_X3,c,c,1);
	cudaDeviceSynchronize();

	dt *d_W1,*d_W2,*d_W3; 
	cudaMalloc((void**)&d_W1,sizeof(dt)*a);
	cudaMalloc((void**)&d_W2,sizeof(dt)*b);
	cudaMalloc((void**)&d_W3,sizeof(dt)*c);
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	cusolverEigRange_t range = CUSOLVER_EIG_RANGE_ALL;
	int meig1=a; int meig2=b; int meig3=c;
	// turn X2X2 to eigvectores and view as U2
	cusolverDnDsyevdx_bufferSize(cusolverH,jobz,range,uplo,b,d_X2_X2,b,0.0,1e06,1,b,&meig2,d_W2,&lwork);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnDsyevdx(cusolverH,jobz,range,uplo,b,d_X2_X2,b,0.0,1e06,1,b,&meig2,d_W2,d_work,lwork,devInfo);
	d_U2=d_X2_X2+(b-r2)*b;
	cudaMemcpy(&infogpu,devInfo,sizeof(int),cudaMemcpyDeviceToHost);
	cout<<infogpu<<endl;
	cudaDeviceSynchronize();
	//cout<<"second vectors"<<endl; printTensor(d_X2_X2+(b-r2)*b,b,r2,1);
	
	// turn X3X3 to eigvectores and view as U3
	cusolverDnDsyevdx_bufferSize(cusolverH,jobz,range,uplo,c,d_X3_X3,c,0.0,1e06,1,c,&meig3,d_W3,&lwork);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnDsyevdx(cusolverH,jobz,range,uplo,c,d_X3_X3,c,0.0,1e06,1,c,&meig3,d_W3,d_work,lwork,devInfo);
	d_U3=d_X3_X3+(c-r3)*c;
	cudaMemcpy(&infogpu,devInfo,sizeof(int),cudaMemcpyDeviceToHost);
	cout<<infogpu<<endl;
	cudaDeviceSynchronize();

	cudaFree(d_X2_X2);
	cudaFree(d_X3_X3);
	cudaFree(d_X2);
	cudaFree(d_Idemat);

	// space for internal val
	dt *d_A1,*d_A2;
	dt *d_AA2;
	cudaMalloc((void**)&d_AA2,sizeof(dt)*a*a);
	cudaMalloc((void**)&d_A1,sizeof(dt)*a*r2*c);
	cudaMalloc((void**)&d_A2,sizeof(dt)*a*r2*r3);
	dt *d_B1,*d_B2,*d_B2T;
	dt *d_BB2;
	cudaMalloc((void**)&d_BB2,sizeof(dt)*b*b);
	cudaMalloc((void**)&d_B1,sizeof(dt)*r1*b*c);
	cudaMalloc((void**)&d_B2,sizeof(dt)*r1*b*r3);
	cudaMalloc((void**)&d_B2T,sizeof(dt)*r1*b*r3);
	dt *d_C1,*d_C2;
	dt *d_CC2;
	cudaMalloc((void**)&d_CC2,sizeof(dt)*c*c);
	cudaMalloc((void**)&d_C1,sizeof(dt)*a*r2*c);
	cudaMalloc((void**)&d_C2,sizeof(dt)*r1*r2*c);
	int L =1;
for(int i=0;i<L;i++){
	//update U1 a*r1;
	cublasDgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,a,r2,b,&alpha,d_X,a,a*b,d_U2,b,0,&beta,d_A1,a,a*r2,c);
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,a*r2,r3,c,&alpha,d_A1,a*r2,d_U3,c,&beta,d_A2,a*r2);
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,a,r2*r3,&alpha,d_A2,a,d_A2,a,&beta,d_AA2,a);
	// syevd for U1
	cusolverDnDsyevdx_bufferSize(cusolverH,jobz,range,uplo,a,d_AA2,a,0.0,1e06,1,a,&meig1,d_W1,&lwork);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnDsyevdx(cusolverH,jobz,range,uplo,a,d_AA2,a,0.0,1e06,1,a,&meig1,d_W1,d_work,lwork,devInfo);
	cudaMemcpy(&infogpu,devInfo,sizeof(int),cudaMemcpyDeviceToHost);
	cout<<infogpu<<endl;
	cudaDeviceSynchronize();
	d_U1=d_AA2+(a-r1)*a; 
	//cout<<"first vectores"<<endl; printTensor(d_U1,a,r1,1);

	//update U2 b*r2
	cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r1,b*c,a,&alpha,d_U1,a,d_X,a,&beta,d_B1,r1);
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,r1*b,r3,c,&alpha,d_B1,r1*b,d_U3,c,&beta,d_B2,r1*b);
	// d_B2 r1*b*r3 den r1*r1  = b*r1*r3 =>b*c
	cublasDgemmStridedBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,b,r1,r1,&alpha,d_B2,r1,r1*b,d_R1den,r1,0,&beta,d_B2T,b,b*r1,r3);
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,b,r1*r3,&alpha,d_B2T,b,d_B2T,b,&beta,d_BB2,b);
	// syevd for U2
	cusolverDnDsyevdx_bufferSize(cusolverH,jobz,range,uplo,b,d_BB2,b,0.0,1e06,1,b,&meig2,d_W2,&lwork);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnDsyevdx(cusolverH,jobz,range,uplo,b,d_BB2,b,0.0,1e06,1,b,&meig2,d_W2,d_work,lwork,devInfo);
	cudaMemcpy(&infogpu,devInfo,sizeof(int),cudaMemcpyDeviceToHost);
	cout<<infogpu<<endl;
	cudaDeviceSynchronize();
	d_U2=d_BB2+(b-r2)*b; 
	//cout<<"second vectores"<<endl; printTensor(d_U2,b,r2,1);

	//update U3 c*r3
	cublasDgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,a,r2,b,&alpha,d_X,a,a*b,d_U2,b,0,&beta,d_C1,a,a*r2,c);
	cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r1,r2*c,a,&alpha,d_U1,a,d_C1,a,&beta,d_C2,r1);
	cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,c,c,r2*r1,&alpha,d_C2,r1*r2,d_C2,r1*r2,&beta,d_CC2,c);
	// syevd for U3
	cusolverDnDsyevdx_bufferSize(cusolverH,jobz,range,uplo,c,d_CC2,c,0.0,1e06,1,c,&meig3,d_W3,&lwork);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	cusolverDnDsyevdx(cusolverH,jobz,range,uplo,c,d_CC2,c,0.0,1e06,1,c,&meig3,d_W3,d_work,lwork,devInfo);
	cudaMemcpy(&infogpu,devInfo,sizeof(int),cudaMemcpyDeviceToHost);
	cout<<infogpu<<endl;
	cudaDeviceSynchronize();
	d_U3=d_CC2+(c-r3)*c; 
	//cout<<"third vectores"<<endl; printTensor(d_U3,c,r3,1);
	cudaDeviceSynchronize();
}
	cudaFree(d_A1);	cudaFree(d_A2); cudaFree(d_AA2);
	cudaFree(d_B1); cudaFree(d_B2); cudaFree(d_B2T); cudaFree(d_BB2);
	cudaFree(d_C1); cudaFree(d_C2); cudaFree(d_CC2);
	cudaFree(d_W1); cudaFree(d_W2); cudaFree(d_W3);
	
	// then solve X x1U1 x2U2 x3U3,we need extra two intenal vals and core to store last result 
	// a*b*c  a*r1  b*r2  c*r3
	// X x1U1' =U1'*X1  X1 can obtain direct store as X
	dt *d_XU1,*d_XU1U2,*d_core;
	cudaMalloc((void**)&d_XU1,sizeof(dt)*r1*b*c);
	cudaMalloc((void**)&d_XU1U2,sizeof(dt)*r1*r2*c);
	cudaMalloc((void**)&d_core,sizeof(dt)*r1*r2*r3);
	//X X1 U1 a*b*c a*r1
	cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,r1,b*c,a,&alpha,d_U1,a,d_X,a,&beta,d_XU1,r1);
	//cout<<"XU1"<<endl; printTensor(d_XU1,r1,b*c,1);
	//XU1*U2 r1*b *c  b*r2
	cublasDgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,r1,r2,b,&alpha,d_XU1,r1,r1*b,d_U2,b,0,&beta,d_XU1U2,r1,r1*r2,c);
	//cout<<"XU1U2"<<endl; printTensor(d_XU1U2,r1,r2*c,1);
	//XU1U2*U3'  r1*r2*r3 c*r3
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,r1*r2,r3,c,&alpha,d_XU1U2,r1*r2,d_U3,c,&beta,d_core,r1*r2);
	//cout<<"core"<<endl; printTensor(d_core,1,r1*r2*r3,1);
	cudaDeviceSynchronize();

	//recover by X = core X1U1 X2U2 X3U3
	// r1*r2*r3  a*r1 ,b*r2 ,c*r3
	dt *d_coreU1,*d_coreU1U2,*d_rec;
	cudaMalloc((void**)&d_coreU1,sizeof(dt)*a*r2*r3);
	cudaMalloc((void**)&d_coreU1U2,sizeof(dt)*a*b*r3);
	cudaMalloc((void**)&d_rec,sizeof(dt)*a*b*c);
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,a,r2*r3,r1,&alpha,d_U1,a,d_core,r1,&beta,d_coreU1,a);
	//cout<<"coreU1"<<endl; printTensor(d_coreU1,a,r2,r3);
	//a*r2*r3  b*r2 coreU1U2
	cublasDgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b,r2,&alpha,d_coreU1,a,a*r2,d_U2,b,0,&beta,d_coreU1U2,a,a*b,r3);
	//cout<<"coreU1U2"<<endl; printTensor(d_coreU1U2,a,b,r3);
	//a*b*r3 c*r3 rec a*b*c
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a*b,c,r3,&alpha,d_coreU1U2,a*b,d_U3,c,&beta,d_rec,a*b);
//	cout<<"rec"<<endl; printTensor(d_rec,a,b,c);

	cublasDaxpy(handle,a*b*c,&alpha1,d_X,1,d_rec,1);
	cublasDnrm2(handle,a*b*c,d_rec,1,&sh);
	cout<<"shang "<<endl; cout<<sh<<endl;
	cublasDnrm2(handle,a*b*c,d_X,1,&xia);
	cudaDeviceSynchronize();
	cout<<"xia "<<endl; cout<<xia<<endl;
	cout<<"error rate "<<sh/xia<<endl;

	cudaFree(d_coreU1);
	cudaFree(d_coreU1U2);
	cudaFree(d_rec);

	// transfer result to Host
	cudaMemcpyAsync(core,d_core,sizeof(dt)*r1*r2*r3,cudaMemcpyDeviceToHost,0);
	cudaMemcpyAsync(U1,d_U1,sizeof(dt)*a*r1,cudaMemcpyDeviceToHost,0);
	cudaMemcpyAsync(U2,d_U2,sizeof(dt)*b*r2,cudaMemcpyDeviceToHost,0);
	cudaMemcpyAsync(U3,d_U3,sizeof(dt)*c*r3,cudaMemcpyDeviceToHost,0);

	cudaDeviceSynchronize();
	cublasDestroy(handle);
	cusolverDnDestroy(cusolverH);
	cudaFree(d_U1); cudaFree(d_U3); cudaFree(d_U2);
	cudaFree(d_XU1);
	cudaFree(d_XU1U2);
	cudaFree(d_core);
	cudaFree(d_work);
	cudaFree(devInfo);
	cudaFree(d_X);
	cudaFree(d_X2);
	cudaFree(d_Idemat);
	cudaFree(d_R1den);
	cudaDeviceReset();
}

