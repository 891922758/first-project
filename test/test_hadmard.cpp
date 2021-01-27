
#include "cudecompose.h" 

using namespace std;

void printTensor(dt *A,int a,int b,int c){
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

int main(int argc,char *argv[]){

		long a = 2;
		long b = 2;
		dt *A,*B,*C;
		cout<<a<<endl;
		cudaHostAlloc((void**)&A,sizeof(dt)*a*b,0);
		cudaHostAlloc((void**)&B,sizeof(dt)*a*b,0);
		cudaHostAlloc((void**)&C,sizeof(dt)*a*b,0);
		for(long i = 0;i<a*b;i++){
			A[i] = rand()*0.1/(RAND_MAX);		//initial Tensor A
		}
		for(long i = 0;i<a*b;i++){
			B[i] = rand()*0.1/(RAND_MAX);		//initial Tensor A
		}
		for(long i = 0;i<a*b;i++){
			C[i] = rand()*0.1/(RAND_MAX);		//initial Tensor A
		}
printTensor(A,a,b,1);
printTensor(B,a,b,1);
		dt *d_A;
		dt *d_B;
		dt *d_C;
		cudaMalloc((void **)&d_A,sizeof(dt)*a*b);
		cudaMalloc((void **)&d_B,sizeof(dt)*b*b);
		cudaMalloc((void **)&d_C,sizeof(dt)*a*b);
		cudaMemcpyAsync(d_A,A,sizeof(dt)*a*b,cudaMemcpyHostToDevice,0);
		cudaMemcpyAsync(d_B,B,sizeof(dt)*b*a,cudaMemcpyHostToDevice,0);
		cudaDeviceSynchronize();

		hadmard(d_A,d_B,d_C,a,b);
		
		cudaFree(d_A);
		cudaFree(d_B);
		cudaMemcpyAsync(C,d_C,sizeof(dt)*a*b,cudaMemcpyDeviceToHost,0);
		cudaDeviceSynchronize();
printTensor(C,a,b,1);
		cudaFree(d_C);

		cudaFreeHost(A);
		cudaFreeHost(B);
		cudaFreeHost(C);
	
		return 0;
}
	
