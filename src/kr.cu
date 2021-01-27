#include <cudecompose.h>
 
void kr(dt *d_A,dt *d_B,dt *d_AkrB,long a,long b,long r){

	dt beta = 0.0;
	dt alpha = 1.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,b,a,1,&alpha,d_B,b,b,d_A,a,a,&beta,d_AkrB,b,b*a,r);
	cublasDestroy(handle);	
}

void printTensor(dt *A,long a,long b,long c){
	dt *h_A;
	cudaHostAlloc((void**)&h_A,sizeof(dt)*a*b*c,0);
	cudaMemcpyAsync(h_A,A,sizeof(dt)*a*b*c,cudaMemcpyDeviceToHost,0);
	cudaDeviceSynchronize();

	for(long i = 0;i<c;i++){
		for(long j = 0;j<a;j++){
			for(long k =0;k<b;k++){
				cout<<h_A[i*a*b+k*a+j]<<"  ";
			}
			cout<<endl;
		}
		cout<<"-----------------------------------"<<endl;
	}
	cout<<endl;
	cudaFreeHost(h_A);
}


