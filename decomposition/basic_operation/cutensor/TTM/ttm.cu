#include "head.h"

__global__  void floattohalf(dt *AA,half *BB,long m){
	long i = blockIdx.x*blockDim.x+threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	if(i<m){
		BB[i]=__float2half(AA[i]);
		i+=temp;
	}
	__syncthreads();
}

void f2h(dt *A,half *B,long num){
	dim3 threads(512,1,1);
	dim3 blocks((num+512-1)/512,1,1);	
	floattohalf<<<blocks,threads>>>(A,B,num);
}
void printHostTensor(dt *A,int a,int b,int c){
	for(int i = 0;i<c;i++){
		for(int j = 0;j<a;j++){
			for(int k =0;k<b;k++){
				cout<<A[i*a*b+k*a+j]<<"  ";
			}
			cout<<endl;
		}
		cout<<"-----------------------------------"<<endl;
	}
	cout<<endl;
}
void printTensor(dt *A,int a,int b,int c){
	dt *h_A;
	cudaHostAlloc((void**)&h_A,sizeof(dt)*a*b*c,0);
	cudaMemcpyAsync(h_A,A,sizeof(dt)*a*b*c,cudaMemcpyDeviceToHost,0);
	cudaDeviceSynchronize();

	for(int i = 0;i<c;i++){
		for(int j = 0;j<a;j++){
			for(int k =0;k<b;k++){
				cout<<h_A[i*a*b+k*a+j]<<"  ";
			}
			cout<<endl;
		}
		cout<<"-----------------------------------"<<endl;
	}
	cout<<endl;
	cudaFreeHost(h_A);
}

double iiplab_mode(dt *X, dt *U1,dt *U2,dt *U3, long a, long b, long c, long d){

	double time1 = 0.0;
	double time2 = 0.0;
	double time3 = 0.0;
	double time11 = 0.0;
	double time22 = 0.0;
	double time33 = 0.0;
	dt alpha = 1.0;
	dt beta = 0.0;

	cublasHandle_t handle;
	cublasCreate(&handle);
//	cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);
	cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH);
	
	dt *d_X, *d_U1, *d_U2,*d_U3,*d_XU1,*d_XU2,*d_XU3;
	cudaMalloc((void**)&d_X,sizeof(dt)*a*b*c);
	cudaMalloc((void**)&d_U1,sizeof(dt)*d*a);
	cudaMalloc((void**)&d_U2,sizeof(dt)*d*b);
	cudaMalloc((void**)&d_U3,sizeof(dt)*d*c);
	cudaMalloc((void**)&d_XU1,sizeof(dt)*d*b*c);
	cudaMalloc((void**)&d_XU2,sizeof(dt)*a*d*c);
	cudaMalloc((void**)&d_XU3,sizeof(dt)*a*b*d);
	cudaMemcpyAsync(d_X,X,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice,0);
	cudaDeviceSynchronize();
	cudaMemcpyAsync(d_U1,U1,sizeof(dt)*d*a,cudaMemcpyHostToDevice,0);
	cudaDeviceSynchronize();
	cudaMemcpyAsync(d_U2,U2,sizeof(dt)*d*b,cudaMemcpyHostToDevice,0);
	cudaDeviceSynchronize();
	cudaMemcpyAsync(d_U3,U3,sizeof(dt)*d*c,cudaMemcpyHostToDevice,0);
	cudaDeviceSynchronize();

	half *h_X,*h_U1,*h_U2,*h_U3;
	cudaMalloc((void**)&h_X,sizeof(half)*a*b*c);
	cudaMalloc((void**)&h_U1,sizeof(half)*a*d);
	cudaMalloc((void**)&h_U2,sizeof(half)*b*d);
	cudaMalloc((void**)&h_U3,sizeof(half)*c*d);
	cudaDeviceSynchronize();

	f2h(d_X,h_X,a*b*c);
	f2h(d_U1,h_U1,a*d);
	f2h(d_U2,h_U2,b*d);
	f2h(d_U3,h_U3,c*d);
	cudaDeviceSynchronize();

	int L = 10;
	//test TTM
	GPUTimer timer;
	for(int iter = 0;iter<L;++iter){
		// warm up
		cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,d,b*c,a,&alpha,d_U1,d,d_X,a,&beta,d_XU1,d);
		cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,d,b,&alpha,d_X,a,a*b,d_U2,d,0,&beta,d_XU2,a,a*d,c);
		cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a*b,d,c,&alpha,d_X,a*b,d_U3,d,&beta,d_XU3,a*b);

		// begin
    		timer.start();
		cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,d,b*c,a,&alpha,h_U1,CUDA_R_16F,d,h_X,CUDA_R_16F,a,&beta,d_XU1,CUDA_R_32F,d,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		cudaDeviceSynchronize();
		time1 = time1+timer.seconds();

	//	cout<<"opt-d_XU1"<<endl;
	//	printTensor(d_XU1+2*2,3,3,2);
	//	printTensor(d_XU1,d,b,c);

    		timer.start();
		cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,d,b*c,a,&alpha,d_U1,d,d_X,a,&beta,d_XU1,d);
		cudaDeviceSynchronize();
		time11 = time11+timer.seconds();
/*		cout<<"unopt-d_XU1"<<endl;
		printTensor(d_XU1,d,b,c);
		printTensor(d_XU1+2*2,3,3,2);
	*/
    		timer.start();
		cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,d,b,&alpha,h_X,CUDA_R_16F,a,a*b,h_U2,CUDA_R_16F,d,0,&beta,d_XU2,CUDA_R_32F,a,a*d,c,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		time2 = time2+timer.seconds();
	//	cout<<"opt-d_XU2"<<endl;
	//	printTensor(d_XU2,a,d,c);
	//	printTensor(d_XU2+2*2,3,3,2);

    		timer.start();
		cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,d,b,&alpha,d_X,a,a*b,d_U2,d,0,&beta,d_XU2,a,a*d,c);
		cudaDeviceSynchronize();
		time22 = time22+timer.seconds();
	/*	cout<<"unopt-d_XU2"<<endl;
	//	printTensor(d_XU2,a,d,c);
		printTensor(d_XU2+2*2,3,3,2);
    	*/	
		timer.start();
		cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,a*b,d,c,&alpha,h_X,CUDA_R_16F,a*b,h_U3,CUDA_R_16F,d,&beta,d_XU3,CUDA_R_32F,a*b,CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		time3 = time3+timer.seconds();
	//	cout<<"opt-d_XU3"<<endl;
	//	printTensor(d_XU3+2*2,3,3,2);
	//	printTensor(d_XU3,a,b,d);

		timer.start();
		cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a*b,d,c,&alpha,d_X,a*b,d_U3,d,&beta,d_XU3,a*b);
		cudaDeviceSynchronize();
		time33 = time33+timer.seconds();
/*		cout<<"upopt-d_XU3"<<endl;
//		printTensor(d_XU3,a,b,d);
		printTensor(d_XU3+2*2,3,3,2);
*/
		if(iter == L-1){
			cout<<"ttm1-time = "<<time11/L<<"ms"<<endl;
			cout<<"ttm1-tensor-time = "<<time1/L<<"ms"<<endl;
			cout<<"ttm2-time = "<<time22/L<<"ms"<<endl;
			cout<<"ttm2-tensor-time = "<<time2/L<<"ms"<<endl;
			cout<<"ttm3-time = "<<time33/L<<"ms"<<endl;
			cout<<"ttm3-tensor-time = "<<time3/L<<"ms"<<endl;
		}
	}

	cublasDestroy(handle);
	cudaFree(d_XU1);
	cudaFree(d_XU2);
	cudaFree(d_XU3);
	cudaFree(d_U1);
	cudaFree(d_U2);
	cudaFree(d_U3);
	cudaFree(d_X);
	cudaFree(h_U1);
	cudaFree(h_U2);
	cudaFree(h_U3);
	cudaFree(h_X);
//	cudaDeviceReset();
	return 1;
}
