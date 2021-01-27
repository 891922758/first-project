/*************************************************************************
	> File Name: tensor.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2018年09月09日 星期日 20时28分47秒
 ************************************************************************/

#include "opera.h"
#include <fstream>
typedef float dt;
using namespace std;
int main(int argc,char *argv[]){

	int r1,r2,r3;
for(int i = 140;i<180;i=i+10){
	int a = i;
	int b = a;
	int c = a;
	if(a<10){
		r1 = 1;
		r2 = 1;
		r3 = 1;
	}else{
		 r1 = a/10;			//assume the core size of r1*r2*r3
		 r2 = b/10;
		 r3 = c/10;
		
	}
	dt *A = new dt[a*b*c]();   //Tensor to be decom
	dt *core = new dt[r1*r2*r3]();

	for(int i = 0;i<a*b*c;i++){
		A[i] = rand()*0.1/(RAND_MAX);		//initial Tensor A
	}

//	printTensor(A,a,b,c);

	dt *U1 = new dt[a*r1]();	//a*r1
	dt *U2 = new dt[b*r2]();	//b*r2
	dt *U3 = new dt[c*r3]();	//c*r3  3 mat factors

	clock_t start,end;
	start = clock();
	HOSVD(A,core,U1,U2,U3,a,b,c);	//function for tuckey 
	end = clock();
	cout<<a<<"  "<<endl;
	cout<<(double)(end-start)/CLOCKS_PER_SEC<<"s"<<endl;
//	printTensor(core,r1,r2,r3);
	
	ofstream outfile("ttime.txt",ios::app);
	outfile<<a<<"*"<<a<<"*"<<a<<"  ";
	outfile<<(double)(end-start)/CLOCKS_PER_SEC<<"s"<<endl;
	outfile.close();

	delete[] A; A = nullptr;
	delete[] U1; U1 = nullptr;
	delete[] U2; U2 = nullptr;
	delete[] U3; U3 = nullptr;
	delete[] core; core = nullptr;
}

	return 0;
}
