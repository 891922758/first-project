
#include "cudecompose.h"

__global__ void tensorTomat1(dt *T1,dt *T2,long m,long n,long k ){
	long i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<m*n*k){
		long tube = i/(m*n);
		long row = (i-tube*(m*n))%m;
		long col = (i-tube*(m*n))/m;
		T2[tube*m*n+col*m+row] = T1[tube*m*n+col*m+row];
	}
	__syncthreads();
	
}

__global__ void tensorTomat2(dt *T1,dt *T2,long m,long n,long k){
	long i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<m*n*k){
		long tube = i/(m*n);
		long row = (i-tube*(m*n))%m;
		long col = (i-tube*(m*n))/m;
		T2[tube*m*n+row*n+col] = T1[tube*m*n+col*m+row];
	}
    __syncthreads();
}

__global__ void tensorTomat3(dt *T1,dt *T2,long m,long n,long k){
	long i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<m*n*k){
		long tube = i/(m*n);
		long row = (i-tube*(m*n))%m;
		long col = (i-tube*(m*n))/m;
		T2[k*(col*m+row)+tube] = T1[tube*m*n+col*m+row];
	}
    __syncthreads();
}

void t2m(dt *d_T1,dt *d_T2,long flag,long m,long n,long k){
	
	dim3 thread(512,1,1);
	dim3 block((m*n*k+512-1)/512,1,1);
	if(flag == 1){
		tensorTomat1<<<block,thread>>>(d_T1,d_T2,m,n,k);
	}else if(flag == 2){
		tensorTomat2<<<block,thread>>>(d_T1,d_T2,m,n,k);
	}else if(flag == 3){
		tensorTomat3<<<block,thread>>>(d_T1,d_T2,m,n,k);
	}else {
		cout<<"no maore than 3"<<endl;
	}

}
