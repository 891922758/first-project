/*************************************************************************
	> File Name: head.h
	> Author: hanlu
	> Mail: 130210201@shu.edu.cn 
	> Created Time: 2019年04月23日 星期二 15时33分02秒
 ************************************************************************/
#ifndef GUARD_cudecompose_h
#define GUARD_cudecompose_h
#include<iostream>
#include<fstream>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cusolverDn.h>
#include<curand.h>
#include <time.h>
#include <cuda_fp16.h>
#include <stdio.h>
using namespace std;

typedef float dt;

extern __global__ void hardm(dt *M,dt *N,dt *res,long m,long n);
void hadmard(dt *d_A,dt *d_B,dt *d_C,long a,long b);
void kr(dt *d_A,dt *d_B,dt *d_AkrB,long a,long b,long r);
void tensor_product(dt *d_A,dt *d_B,dt *d_C,long a,long b,long c,long d);
void printTensor(dt *A,long a,long b,long c);
extern __global__  void floattohalf(dt *AA,half *BB,long m);
void f2h(dt *A,half *B,long num);
void ttm(dt *d_X, dt *d_U,dt *d_XU, long flag, long a, long b, long c, long d);
void mttkrp(dt *d_X, dt *left,dt *d_right, dt *d_XU, long flag, long m, long n, long k, long r);
extern __global__ void tensorTomat1(dt *T1,dt *T2,long m,long n,long k );
extern __global__ void tensorTomat2(dt *T1,dt *T2,long m,long n,long k );
extern __global__ void tensorTomat3(dt *T1,dt *T2,long m,long n,long k );
void t2m(dt *d_T1,dt *d_T2,long flag,long m,long n,long k);
extern __global__ void t_t(dt *M,dt *N,long m);
void tminust(dt *d_left,dt *d_right,long m);
void rse(dt *d_X,dt *d_X1,long m,dt *error);
void gencp(dt *d_rec,dt *d_A,dt *d_B,dt *d_C,long a,long b,long c,long r);
void gentucker(dt *d_rec,dt *d_core, dt *d_A,dt *d_B,dt *d_C,long a,long b,long c,long r1,long r2,long r3);
void cp_als(dt *d_X,dt *d_A,dt *d_B,dt *d_C,long a,long b,long c,long r);
void tucker_hosvd(dt *d_X,dt *d_core,dt *d_U1,dt *d_U2,dt *d_U3,long a,long b,long c,long r1,long r2,long r3);
void cp_tensorcore(dt *d_X,dt *d_A,dt *d_B,dt *d_C,long a,long b,long c,long r);
void tucker_tensorcore(dt *d_X,dt *d_core,dt *d_U1,dt *d_U2,dt *d_U3,long a,long b,long c,long r1,long r2,long r3);

extern __global__ void initIdeMat(dt *AA,long m);
void initMat(dt *d_A,long m);
extern __global__ void hardm3(dt *M,dt *N,dt *K, dt *res,long  m, long n);
void hadmard3(dt *d_A,dt *d_B,dt *d_C,dt *d_ABC,long a,long b);

void cp_als4(dt *d_X,dt *d_A,dt *d_B,dt *d_C, dt *d_D,long a,long b,long c,long d,long r);
void tucker_hosvd4(dt *d_X,dt *d_core,dt *d_U1,dt *d_U2,dt *d_U3,dt *d_U4,long a,long b,long c,long d,long r1,long r2,long r3,long r4);

void gencp4(dt *d_T,dt *d_AA,dt *d_BB,dt *d_CC,dt *d_DD,long a,long b,long c,long d,long r);
void gentucker4(dt *d_T,dt *d_G, dt *d_A,dt *d_B,dt *d_C,dt *d_D,long a,long b,long c,long d,long r1,long r2,long r3,long r4);
extern __global__ void elepro3(dt *AA,dt *BB,dt *CC,dt *DD,int m);
extern __global__ void initIdeMat(dt *AA,int m);
extern __global__ void elemin(dt *A,dt *B, long n);

#endif
