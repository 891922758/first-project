
/*************************************************************************
	> File Name: ttm1.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2019年04月23日 星期二 15时53分16秒
 ************************************************************************/
#include "cudecompose.h"

int main(int argc,char *argv[]){
		const long a = 3;
		const long b = 3;
		const long c = 3;
		const long d = 2;

		dt *X ,*U1,*U2,*U3;
		cudaHostAlloc((void**)&X,sizeof(dt)*a*b*c,0);
		cudaHostAlloc((void**)&U1,sizeof(dt)*d*a,0);
		cudaHostAlloc((void**)&U2,sizeof(dt)*d*b,0);
		cudaHostAlloc((void**)&U3,sizeof(dt)*d*c,0);
		srand(2);

		for(int number = 0; number < a*b*c; ++number){
			X[number] = (rand()*0.1/(RAND_MAX) - 0.5);
		}
		for(int number = 0; number < d*a; ++number){
			U1[number] = (rand()*0.1/(RAND_MAX) - 0.5);
		}
		for(int number = 0; number < d*b; ++number){
			U2[number] = (rand()*0.1/(RAND_MAX) - 0.5);
		}
		for(int number = 0; number < d*c; ++number){
			U3[number] = (rand()*0.1/(RAND_MAX) - 0.5);
		}
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
	printTensor(d_X,a,b,c);
	printTensor(d_U1,d,a,1);
//	printTensor(d_U2,d,b,1);
//	printTensor(d_U3,d,c,1);

	ttm(d_X,d_U1,d_XU1,1,a,b,c,d);
	printTensor(d_XU1,d,b,c);
//	ttm(d_X,d_U2,d_XU2,2,a,b,c,d);
//	printTensor(d_XU2,a,d,c);
//	ttm(d_X,d_U3,d_XU3,3,a,b,c,d);
//	printTensor(d_XU3,a,b,d);

	cudaFree(d_XU1);
	cudaFree(d_XU2);
	cudaFree(d_XU3);
	cudaFree(d_U1);
	cudaFree(d_U2);
	cudaFree(d_U3);
	cudaFree(d_X);
		
	cudaFreeHost(X);
	cudaFreeHost(U1);
	cudaFreeHost(U2);
	cudaFreeHost(U3);

	return 0;
}

