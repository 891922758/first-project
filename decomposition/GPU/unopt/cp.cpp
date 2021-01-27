/*************************************************************************
	> File Name: cp.cpp
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2018年09月22日 星期六 19时48分28秒
 ************************************************************************/

#include "opera.h"

typedef float dt;
using namespace std;

int main(int argc,char *argv[]){
	clock_t start,end;

for(int i = 140;i<=170;i=i+10){
    int a = i;
    int b = a;
    int c = a;
	int r;
	if(a<10){
		r = 1;
	}else{
		r = a*0.1;
	}

	dt *X = new dt[a*b*c]();
	for(int i = 0;i<a*b*c;i++){
		X[i] = rand()*0.1/(RAND_MAX);
	}
//	printTensor(X,a,b,c);
	dt *A = new dt[a*r]();
	dt *B = new dt[b*r]();
	dt *C = new dt[c*r]();
	for(int i = 0;i<a*r;i++){
		A[i] = rand()*0.1/(RAND_MAX);
	}
//	printTensor(A,a,r,1);
	for(int i = 0;i<b*r;i++){
		B[i] = rand()*0.1/(RAND_MAX);
	}
//	printTensor(B,b,r,1);
	for(int i = 0;i<c*r;i++){
		C[i] = rand()*0.1/(RAND_MAX);
	}
//	printTensor(C,c,r,1);
	start = clock();
	cp_als(X,A,B,C,a,b,c,r);
	end = clock();
    	
	cout<<a<<"*"<<a<<"*"<<a<<"  "; 
	cout<<(double)(end-start)/CLOCKS_PER_SEC<<"s"<<endl;

	ofstream outfile("ctime.txt",ios::app);
	outfile<<a<<"*"<<a<<"*"<<a<<"  ";
	outfile<<(double)(end-start)/CLOCKS_PER_SEC<<"s"<<endl;
	outfile.close();

	delete[] X;X = nullptr;
	delete[] A;A = nullptr;
	delete[] B;B = nullptr;
	delete[] C;C = nullptr;
}
	return 0;
}
