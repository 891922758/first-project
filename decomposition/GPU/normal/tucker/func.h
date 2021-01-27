/*************************************************************************
	> File Name: func.h
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2019年04月23日 星期二 19时00分17秒
 ************************************************************************/
#ifndef GUARD_func_h
#define GUARD_func_h

typedef float dt;
void tucker_tensorcore(dt *X,dt *core,dt *U1,dt *U2,dt *U3,long a,long b,long c,long r1,long r2,long r3);
__global__  void floattohalf(dt *AA,half *BB,long m);
void f2h(dt *A,half *B,long num);
__global__ void elemin(dt *A,dt *B, long n);
__global__ void tensorToMode2(dt *T1,dt *T2,int m,int n,int k);
void cp_als(dt *X,dt *A,dt *B,dt *C,long a,long b,long c,long r);
void cp_als4(dt *X,dt *A,dt *B,dt *C,dt *D,long a,long b,long c,long d,long r);
void printTensor(dt *d_des,long m,long n,long l);
__global__ void elepro(dt *AA,dt *BB,dt *CC,int m);
__global__ void elepro3(dt *AA,dt *BB,dt *CC,dt *DD,int m);
__global__ void initIdeMat(dt *AA,int m);
void tucker_hosvd(dt *X,dt *core,dt *U1,dt *U2,dt *U3,long a,long b,long c,long r1,long r2,long r3);
void tucker_hosvd4(dt *X,dt *core,dt *U1,dt *U2,dt *U3,dt *U4,long a,long b,long c,long d,long r1,long r2,long r3,long r4);
void gencpTensor(dt *T,long a,long b,long c,long r);
void gencpTensor4(dt *T,long a,long b,long c,long d,long r);
void gentuTensor(dt *T,long a,long b,long c,long r1,long r2,long r3);
void gentuTensor4(dt *T,long a,long b,long c,long d,long r1,long r2,long r3,long r4);
void tucker_hooi(dt *X,dt *core,dt *U1,dt *U2,dt *U3,long a,long b,long c,long r1,long r2,long r3);

#endif
