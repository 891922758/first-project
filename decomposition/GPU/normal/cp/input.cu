
#include "head.h"

int main(int argc,char *argv[]){

for(int i =160;i<=1300;i=i+160){
	clock_t t1,t2;
	double times=0.0;

	const long a = i;
	const long b = a;
	const long c = a;
	const long r = (int)(a*0.1);
	dt *X,*A,*B,*C;
	cudaHostAlloc((void**)&X,sizeof(dt)*a*b*c,0);
	cudaHostAlloc((void**)&A,sizeof(dt)*a*r,0);
	cudaHostAlloc((void**)&B,sizeof(dt)*b*r,0);
	cudaHostAlloc((void**)&C,sizeof(dt)*c*r,0);
	gencpTensor(X,a,b,c,r);

	t1=clock();
	cp_tensorcore(X,A,B,C,a,b,c,r);
	t2=clock();
	times = (double)(t2-t1)/CLOCKS_PER_SEC;
	cout<<i<<"   "<<times<<"s"<<endl;
/*
	// recover to X3' which is same to X
	// X3'= (BkrA)*C' 
	dt alpha = 1.0;
	dt beta = 0.0;
	dt sh=0.0;
	dt xia=1.0;
	dim3 threads(512,1,1);
	dim3 block1((r*r+512-1)/512,1,1); //for elepro
	dim3 block2((a*b*c+512-1)/512,1,1);
	cublasHandle_t handle;
	cublasCreate(&handle);
	dt *d_rec,*d_B,*d_C,*d_A,*d_BkrA,*d_X;
	cudaMalloc((void**)&d_rec,sizeof(dt)*b*a*c);
	cudaMalloc((void**)&d_X,sizeof(dt)*b*a*c);
	cudaMalloc((void**)&d_B,sizeof(dt)*b*r);
	cudaMalloc((void**)&d_C,sizeof(dt)*c*r);
	cudaMalloc((void**)&d_A,sizeof(dt)*a*r);
	cudaMalloc((void**)&d_BkrA,sizeof(dt)*b*r*c);
	cudaMemcpyAsync(d_X,X,sizeof(dt)*a*b*c,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_A,A,sizeof(dt)*a*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_B,B,sizeof(dt)*b*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_C,C,sizeof(dt)*r*c,cudaMemcpyHostToDevice,0);
	cudaDeviceSynchronize();

	cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b,1,&alpha,d_A,a,a,d_B,b,b,&beta,d_BkrA,a,a*b,r);
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
	cudaFree(d_B);
	cudaFree(d_A);
	cudaFree(d_C);
	cudaFree(d_BkrA);
	cudaFree(d_X);
*/
	ofstream outfile("cptensorcore.txt",ios::app);
	outfile<<i<<"   "<<times<<"s"<<endl;
	outfile.close();

	cudaFreeHost(X);
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);

}
	return 0;
}

