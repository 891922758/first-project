/*************************************************************************
	> File Name: head.h
	> Author: hanlu
	> Mail: hanlu@shu.edu.cn 
	> Created Time: 2019年04月23日 星期二 15时33分02秒
 ************************************************************************/
#ifndef GUARD_head_h
#define GUARD_head_h
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <cufft.h>

#include <stdlib.h>
#include <stdio.h>
#include <unordered_map>
#include <vector>
#include <cutensor.h>

using namespace std;
typedef float dt;

double iiplab_mode(dt *X, dt *U1,dt *U2,dt *U3, long a, long b, long c, long d);
__global__  void floattohalf(dt *AA,half *BB,long m);
void printTensor(dt *A,int a,int b,int c);
void printHostTensor(dt *A,int a,int b,int c);
void f2h(dt *A,half *B,long num);

#define HANDLE_ERROR(x)                                               \
{ const auto err = x;                                                 \
  if( err != CUTENSOR_STATUS_SUCCESS )                                \
  { printf("Error: %s\n", cutensorGetErrorString(err)); return err; } \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
  if( err != cudaSuccess )                                        \
  { printf("Error: %s\n", cudaGetErrorString(err)); return err; } \
};

struct GPUTimer
{
    GPUTimer() 
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_, 0);
    }

    ~GPUTimer() 
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() 
    {
        cudaEventRecord(start_, 0);
    }

    float seconds() 
    {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        float time;
        cudaEventElapsedTime(&time, start_, stop_);
        return time;
    }
    private:
    cudaEvent_t start_, stop_;
};
#endif
