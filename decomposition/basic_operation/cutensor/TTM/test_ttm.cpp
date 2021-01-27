
/*************************************************************************
	> File Name: ttm1.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2019年04月23日 星期二 15时53分16秒
 ************************************************************************/
#include "head.h"

int main(int argc,char *argv[]){
	for(int i =160;i<=1400;i=i+160){
		const long a = i;
		const long b = i;
		const long c = i;
		const long d = long(a*0.1);
	//	const long d = 3;
		dt *X ,*U1,*U2,*U3;
		cout<<a<<" "<<b<<" "<<c<<" "<<d<<endl;
		cudaHostAlloc((void**)&X,sizeof(dt)*a*b*c,0);
		cudaHostAlloc((void**)&U1,sizeof(dt)*d*a,0);
		cudaHostAlloc((void**)&U2,sizeof(dt)*d*b,0);
		cudaHostAlloc((void**)&U3,sizeof(dt)*d*c,0);

		for(int number = 0; number < a*b*c; ++number){
			X[number] = (rand()*0.1/(RAND_MAX*0.1) - 0.5);
		}
		for(int number = 0; number < d*a; ++number){
			U1[number] = (rand()*0.1/(RAND_MAX*0.1) - 0.5);
		}
		for(int number = 0; number < d*b; ++number){
			U2[number] = (rand()*0.1/(RAND_MAX*0.1) - 0.5);
		}
		for(int number = 0; number < d*c; ++number){
			U3[number] = (rand()*0.1/(RAND_MAX*0.1) - 0.5);
		}
		
		double ttm_time = iiplab_mode(X,U1,U2,U3, a, b, c,d);
		
/*		cout<<i<<endl;
		cout<< ttm_time<< "s;"<<endl;
		
		ofstream outfile("time_iiplab.txt",ios::app);
		
		outfile<<i<<" "<<times_cutensor <<endl;
		outfile.close();
*/
		cudaFreeHost(X);
		cudaFreeHost(U1);
		cudaFreeHost(U2);
		cudaFreeHost(U3);
	}
	return 0;
}

