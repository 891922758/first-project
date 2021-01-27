
#include <stdlib.h>
#include "opera.h"

void SBgemm(dt *A,dt *core,dt *U1,dt *U2,dt *U3,int a,int b,int c){
	int r1 ,r2,r3;
	if(a<10){
		r1 = 1;
		r2 = 1;
		r3 = 1;
	}else{
		r1 = a/10;
		r2 = b/10;
		r3 = c/10;
	}
	//compute A ×1 U1'×2 U2'×3 U3'
	// first compute U1'[X1,X2,X3～～Xc]U2 = temp;   then temp*U3'   
	//  U1 a*r1  U1' r1*a
	// A a*b*c
	// U2 b*r2  U2' r2*b
	// U3 c*r3  U3' r3*c

	dt alpha = 1.0;
	dt beta = 0.0;
	dt *d_A;
	dt *d_U1;
	dt *d_U2;
	dt *d_U3;
	dt *d_temp1;
	dt *d_temp2;
	dt *d_temp3;
	dt *d_temp;
	cudaMalloc((void **)&d_A,a*b*c*sizeof(dt));
	cudaMalloc((void **)&d_U1,a*r1*sizeof(dt));
	cudaMalloc((void **)&d_temp1,sizeof(dt)*b*r1*c);

	cudaMemcpy(d_A,A,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice);
	cudaMemcpy(d_U1,U1,sizeof(dt)*a*r1,cudaMemcpyHostToDevice);
	//cudaMemcpy(d_C,C,sizeof(dt)*a*d*c);

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemmStridedBatched(
			
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_T,
			b,				//row of A C
			r1,				//col of B C
			a,				//col of A ,row of B
			&alpha,
			d_A,
			b,				//leading dimension store A
			b*a,			//step between two mat
		    d_U1,
			r1,
			0,
			&beta,
			d_temp1,
			b,
			b*r1,
			c				//batch number
			);
	//now d_temp1 store the real value col first
//	cudaMemcpy(temp1,d_temp1,sizeof(dt)*b*r1*c,cudaMemcpyDeviceToHost);
//	printTensor(temp1,r1,b,c);
	cudaFree(d_U1);
	cudaFree(d_A);

	cudaMalloc((void **)&d_U2,b*r2*sizeof(dt));
	cudaMalloc((void **)&d_temp2,sizeof(dt)*r1*r2*c);
	cudaMemcpy(d_U2,U2,sizeof(dt)*b*r2,cudaMemcpyHostToDevice);
	cublasSgemmStridedBatched(
			
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			r2,
			r1,
			b,
			&alpha,
			d_U2,
			r2,
			0,
			d_temp1,
			b,
			b*r1,
			&beta,
			d_temp2,
			r2,
			r2*r1,
			c

			);

//	cudaMemcpy(temp2,d_temp2,sizeof(dt)*r1*r2*c,cudaMemcpyDeviceToHost);
	
//	printTensor(temp2,r1,r2,c);
	cudaFree(d_temp1);
	cudaFree(d_U2);

	cudaMalloc((void **)&d_temp3,sizeof(dt)*r1*r2*c);	//mode 3 mat

	// now temp2 store the real value 
	//we will mat3,and the 
	dim3 threads(512,1,1);
	dim3 blocks((r1*r2*c+512-1)/512,1,1);
	mode3tran<<<blocks,threads>>>(d_temp2,d_temp3,r1,r2,c);
//	temp3 = tensor2mat(temp2,r1,r2,c,3);
//	printTensor(temp3,c,r1*r2,1);
	
	cudaFree(d_temp2);

//	cudaMemcpy(d_temp3,temp3,sizeof(dt)*c*r1*r2,cudaMemcpyHostToDevice);

	cudaMalloc((void **)&d_U3,c*r3*sizeof(dt));
	cudaMemcpy(d_U3,U3,sizeof(dt)*c*r3,cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_temp,sizeof(dt)*r1*r2*r3);
	cublasSgemm(
			
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_T,
			r1*r2,
			r3,
			c,
			&alpha,
			d_temp3,
			r1*r2,
			d_U3,
			r3,
			&beta,
			d_temp,
			r1*r2

			);

	//cudaMemcpy(temp,d_temp,sizeof(dt)*r3*r1*r2,cudaMemcpyDeviceToHost);

//	printTensor(temp,r3,r1*r2,1);

	cudaFree(d_U3);
	cudaFree(d_temp3);
    dt *d_core;
    cudaMalloc((void**)&d_core,sizeof(dt)*r1*r2*r3);
    dim3 blocks1((r1*r2*r3+512-1)/512,1,1);
    tran3mode<<<blocks1,threads>>>(d_temp,d_core,r1,r2,r3);
    cudaMemcpy(core,d_core,sizeof(dt)*r1*r2*r3,cudaMemcpyDeviceToHost);
    cudaFree(d_core);

	cudaFree(d_temp);
	cublasDestroy(handle);
	
}

void printTensor(dt *A,int a,int b,int c){
	for(int i = 0;i<c;i++){
		for(int j = 0;j<a;j++){
			for(int k =0;k<b;k++){
				cout<<A[i*a*b+j*b+k]<<"  ";
			}
			cout<<endl;
		}
		cout<<"-----------------------------------"<<endl;
	}
	cout<<endl;
}


void HOSVD(dt *A,dt *core,dt *U1,dt *U2,dt *U3,int a,int b,int c){
	int r1 ,r2,r3;
	if(a<10){
		r1 = 1;
		r2 = 1;
		r3 = 1;
	}else{
		r1 = a/10;
		r2 = b/10;
		r3 = c/10;
	}
	dt *A1 = new dt[a*b*c]();	
	dt *A2 = new dt[a*b*c]();
	dt *A3 = new dt[a*b*c]();	//3 mode tensor to mat

	Btensor2mat(A,A1,A2,A3,a,b,c);

	getvector(A1,U1,a,b*c,r1);
	getvector(A2,U2,b,a*c,r2);
	getvector(A3,U3,c,a*b,r3);
	//compute A ×1 U1'×2 U2'×3 U3'
	// first compute U1'[X1,X2,X3～～Xc]U2 = temp;   then temp*U3'   
	SBgemm(A,core,U1,U2,U3,a,b,c);
	

	delete[] A1; A1 = nullptr;
	delete[] A2; A2 = nullptr;
	delete[] A3; A3 = nullptr;
	}


void Btensor2mat(dt *A,dt *A1,dt *A2,dt *A3,int a,int b,int c){
	
	dt *d_AA;
	dt *d_A1;
	dt *d_A2;
	dt *d_A3;

	cudaMalloc((void **)&d_AA,sizeof(dt)*a*b*c);
	cudaMalloc((void **)&d_A1,sizeof(dt)*a*b*c);

	cudaMemcpy(d_AA,A,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice);

	dim3 threads(512,1,1);
	dim3 blocks(((a*b*c+512-1)/512),1,1);

	mode1tran<<<blocks,threads>>>(d_AA,d_A1,a,b,c);
	cudaMemcpy(A1,d_A1,sizeof(dt)*a*b*c,cudaMemcpyDeviceToHost);
	cudaFree(d_A1);

	cudaMalloc((void **)&d_A2,sizeof(dt)*a*b*c);
	mode2tran<<<blocks,threads>>>(d_AA,d_A2,a,b,c);
	cudaMemcpy(A2,d_A2,sizeof(dt)*a*b*c,cudaMemcpyDeviceToHost);
	cudaFree(d_A2);

	cudaMalloc((void **)&d_A3,sizeof(dt)*a*b*c);
	mode3tran<<<blocks,threads>>>(d_AA,d_A3,a,b,c);

	cudaMemcpy(A3,d_A3,sizeof(dt)*a*b*c,cudaMemcpyDeviceToHost);

	cudaFree(d_AA);
	cudaFree(d_A3);

}

void getvector(dt *A,dt *U,int m,int n,int r){
	//we compute A*A'
	dt *d_A;
	dt *d_AT;
	cudaMalloc((void**)&d_A,sizeof(dt)*m*n);
	cudaMalloc((void**)&d_AT,sizeof(dt)*m*n);
	dt *d_AAT;
	cudaMalloc((void**)&d_AAT,sizeof(dt)*m*m);
	dt alpha = 1.0;
	dt beta = 0.0;
	cudaMemcpy(d_A,A,sizeof(dt)*m*n,cudaMemcpyHostToDevice);
	dim3 threads(512,1,1);
	dim3 blocks((m*n+512-1)/512,1,1);
	transpose<<<blocks,threads>>>(d_A,d_AT,m,n);  // now d_AT n*m
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemm(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			m,
			m,
			n,
			&alpha,
			d_AT,
			m,
			d_A,
			n,
			&beta,
			d_AAT,  //store A*A'
			m
			);
	cublasDestroy(handle);
	cudaFree(d_A);
	cudaFree(d_AT);
// eig
	cusolverDnHandle_t cusolverH = NULL;
	dt *V = new dt[m*m]();
	dt *V1 = new dt[r*m]();
	dt *d_W;
	int *devInfo = NULL;
	dt *d_work = NULL;
	int lwork;
	int info_gpu = 0;
	cusolverDnCreate(&cusolverH);
	cudaMalloc((void**)&devInfo,sizeof(int));
	cudaMalloc((void**)&d_W,sizeof(dt)*m);
	
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	cusolverDnSsyevd_bufferSize(
			cusolverH,
			jobz,
			uplo,
			m,
			d_AAT,
			m,
			d_W,
			&lwork
			);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);

	cusolverDnSsyevd(
			cusolverH,
			jobz,
			uplo,
			m,
			d_AAT,   //store vectors
			m,
			d_W,
			d_work,
			lwork,
			devInfo
			);
	cudaDeviceSynchronize();
	cudaMemcpy(V,d_AAT,sizeof(dt)*m*m,cudaMemcpyDeviceToHost);
//	cudaMemcpy(W,d_W,sizeof(dt)*m,cudaMemcpyDeviceToHost);
/*	cudaMemcpy(&info_gpu,devInfo,sizeof(int),cudaMemcpyDeviceToHost);
	if(info_gpu == 0){
		cout<<"ok"<<endl;
	}else{
		cout<<info_gpu<<endl;
	}
*/
	cudaFree(d_W);
	cudaFree(d_work);
	cudaFree(devInfo);
	cudaFree(d_AAT);
//	printTensor(V,m,m,1);
//	printTensor(W,m,1,1);
	cusolverDnDestroy(cusolverH);
//cudaDeviceReset();
//	printTensor(V,m,m,1);
	for(int i=0;i<r;i++){
		for(int j = 0;j<m;j++){
			V1[i*m+j] = V[i*m+j+m*(m-r)];
			U[j*r+i] = V1[i*m+j];
		}
	}
//	printTensor(U,m,r,1);
		
	delete[] V;V=nullptr;
	delete[] V1;V1=nullptr;

}
void KRao(dt *X,dt *M,dt *N,dt *left,dt *right,int m,int n,int r,int k,int flag){
// m*r  n*r  m*n*r
	
	dt *d_MT;
	dt *d_NT;
	cudaMalloc((void **)&d_MT,sizeof(dt)*m*r);
	cudaMalloc((void **)&d_NT,sizeof(dt)*n*r);
	dim3 threads1(512,1,1);
	dim3 blocks1((m*r+512-1)/512,1,1);
	transpose<<<blocks1,threads1>>>(M,d_MT,m,r);
	dim3 threads2(512,1,1);
	dim3 blocks2((n*r+512-1)/512,1,1);
	transpose<<<blocks2,threads2>>>(N,d_NT,n,r);
	//now d_MT*M  d_NT*N
	dt *d_MTM;
	cudaMalloc((void **)&d_MTM,sizeof(dt)*r*r);

	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemm(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			r,
			r,
			m,
			&alpha,
			M,
			r,
			d_MT,
			m,
			&beta,
			d_MTM,
			r
			);
	cudaFree(d_MT);
	
	dt *d_NTN;
	cudaMalloc((void**)&d_NTN,sizeof(dt)*r*r);
	cublasSgemm(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			r,
			r,
			n,
			&alpha,
			N,
			r,
			d_NT,
			n,
			&beta,
			d_NTN,
			r
			);
	cudaFree(d_NT);

	dim3 threads3(r*r,1,1);
	dim3 blocks3((r*r+512-1)/512,1,1);
	elepro<<<blocks3,threads3>>>(d_MTM,d_NTN,right,r);
	cudaFree(d_MTM);
	cudaFree(d_NTN);
    
	//right is solve the right

	dt *d_dot;
	cudaMalloc((void **)&d_dot,sizeof(dt)*m*n*r);
	dim3 threads(512,1,1);
	dim3 blocks((m*n*r+512-1)/512,1,1);
	krpro<<<blocks,threads>>>(M,N,d_dot,m,n,r);
	//res store the dotpro  bc*a
	dt *d_X_M;
	cudaMalloc((void**)&d_X_M,sizeof(dt)*m*n*k);

	dim3 threads4(512,1,1);
	dim3 blocks4((m*n*k+512-1)/512,1,1);
	if(flag == 1){
		mode1tran<<<blocks4,threads4>>>(X,d_X_M,k,n,m);
	}else if(flag == 2){
		mode2tran<<<blocks4,threads4>>>(X,d_X_M,n,k,m);
	}else{
		mode3tran<<<blocks4,threads4>>>(X,d_X_M,n,m,k);
	}

	// d_X1*d_dot = left
	cublasSgemm(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			r,
			k,
			m*n,
			&alpha,
			d_dot,
			r,
			d_X_M,
			m*n,
			&beta,
			left,
			r
			);
	cublasDestroy(handle);
	cudaFree(d_X_M);
	cudaFree(d_dot);

}



void solve(dt *left,dt *right,dt *res,int r,int m){
	dt *d_work;
	int *d_info;
	int lwork;
	cusolverDnHandle_t handle;
	cusolverDnCreate(&handle);
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	cudaMalloc((void**)&d_info,sizeof(int));

	cusolverDnSpotrf_bufferSize(
			handle,
			uplo,
			r,
			left,
			r,
			&lwork
			);
	cudaMalloc((void**)&d_work,sizeof(dt)*lwork);

	cusolverDnSpotrf(
			handle,
			uplo,
			r,
			left,
			r,
			d_work,
			lwork,
			d_info
			);
	cusolverDnSpotrs(
			handle,
			uplo,
			r,
			m,
			left,
			r,
			right,
			r,
			d_info
			);
	cudaDeviceSynchronize();
//	int info_gpu;
//	cudaMemcpy(&info_gpu,d_info,sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(res,right,sizeof(dt)*m*r,cudaMemcpyDeviceToDevice);
/*	if(info_gpu == 0){
		cout<<"OK"<<endl;
		cout<<endl;
	}
*/

	cudaFree(d_info);
	cudaFree(d_work);
	cusolverDnDestroy(handle);
//	cudaDeviceReset();
}

void cp_als(dt *X,dt *A,dt *B,dt *C,int a,int b,int c,int r){

	dt *d_X,*d_A,*d_B,*d_C;
	cudaMalloc((void**)&d_X,sizeof(dt)*a*b*c);
	cudaMalloc((void**)&d_A,sizeof(dt)*a*r);
	cudaMalloc((void**)&d_B,sizeof(dt)*b*r);
	cudaMalloc((void**)&d_C,sizeof(dt)*c*r);
	cudaMemcpy(d_X,X,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice);
	cudaMemcpy(d_A,A,sizeof(dt)*a*r,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,sizeof(dt)*b*r,cudaMemcpyHostToDevice);
	cudaMemcpy(d_C,C,sizeof(dt)*c*r,cudaMemcpyHostToDevice);
	
	for(int i = 0;i<10;i++){
	dt *d_temp1,*d_temp2,*d_temp3,*d_tem1,*d_tem2,*d_tem3;
	cudaMalloc((void**)&d_temp1,sizeof(dt)*a*r);
	cudaMalloc((void**)&d_temp2,sizeof(dt)*b*r);
	cudaMalloc((void**)&d_temp3,sizeof(dt)*c*r);
	cudaMalloc((void**)&d_tem1,sizeof(dt)*r*r);
	cudaMalloc((void**)&d_tem2,sizeof(dt)*r*r);
	cudaMalloc((void**)&d_tem3,sizeof(dt)*r*r);
	
		KRao(d_X,d_C,d_B,d_temp1,d_tem1,c,b,r,a,1);
		solve(d_tem1,d_temp1,d_A,r,a);     // we get A  

		KRao(d_X,d_C,d_A,d_temp2,d_tem2,c,a,r,b,2);
		solve(d_tem2,d_temp2,d_B,r,b);     // we get B
		
		KRao(d_X,d_B,d_A,d_temp3,d_tem3,b,a,r,c,3);
		solve(d_tem3,d_temp3,d_C,r,c);    //we get C

	cudaFree(d_temp1);
	cudaFree(d_temp2);
	cudaFree(d_temp3);
	cudaFree(d_tem1);
	cudaFree(d_tem2);
	cudaFree(d_tem3);

	}
	cudaMemcpy(C,d_C,sizeof(dt)*c*r,cudaMemcpyDeviceToHost);
	cudaMemcpy(d_A,A,sizeof(dt)*a*r,cudaMemcpyDeviceToHost);
	cudaMemcpy(d_B,B,sizeof(dt)*b*r,cudaMemcpyDeviceToHost);

	cudaFree(d_X);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
