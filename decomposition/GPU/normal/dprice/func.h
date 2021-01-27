/*************************************************************************
	> File Name: func.h
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2019年04月23日 星期二 19时00分17秒
 ************************************************************************/
#ifndef GUARD_func_h
#define GUARD_func_h

typedef double dt;
__global__ void elemin(dt *A,dt *B, long n);
__global__ void tensorToMode2(dt *T1,dt *T2,int m,int n,int k);
void cp_als(dt *X,dt *A,dt *B,dt *C,int a,int b,int c,int r);
void cp_als4(dt *X,dt *A,dt *B,dt *C,dt *D,int a,int b,int c,int d,int r);
void printTensor(dt *d_des,int m,int n,int l);
void printTensor4(dt *d_des,int m,int n,int l,int k);
__global__ void elepro(dt *AA,dt *BB,dt *CC,int m);
__global__ void elepro3(dt *AA,dt *BB,dt *CC,dt *DD,int m);
__global__ void initIdeMat(dt *AA,int m);
void tucker_hosvd(dt *X,dt *core,dt *U1,dt *U2,dt *U3,int a,int b,int c,int r1,int r2,int r3);
void tucker_hosvd4(dt *X,dt *core,dt *U1,dt *U2,dt *U3,dt *U4,int a,int b,int c,int d,int r1,int r2,int r3,int r4);
void gencpTensor(dt *T,int a,int b,int c,int r);
void gencpTensor4(dt *T,int a,int b,int c,int d,int r);
void gentuTensor(dt *T,int a,int b,int c,int r1,int r2,int r3);
void gentuTensor4(dt *T,int a,int b,int c,int d,int r1,int r2,int r3,int r4);
void tucker_hooi(dt *X,dt *core,dt *U1,dt *U2,dt *U3,int a,int b,int c,int r1,int r2,int r3);
void gentuTensor1(dt *T,dt *A,dt *B,dt *C,dt *G,int a,int b,int c,int r1,int r2,int r3);

#endif
