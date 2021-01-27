
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>
using namespace std;

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

int main(int argc, char** argv)
{
	for(int hh = 160;hh<=160;hh = hh+160){
		cout<<hh<<endl;
	    typedef float floatTypeA;
	        typedef float floatTypeC;
		    typedef float floatTypeCompute;

		        cudaDataType_t typeA = CUDA_R_32F;
			    cudaDataType_t typeC = CUDA_R_32F;
			        cudaDataType_t typeCompute = CUDA_R_32F;

				    floatTypeCompute alpha = (floatTypeCompute)1.0f;
				        floatTypeCompute gamma = (floatTypeCompute)1.0f;

					    /**********************
					           * Computing: C_{a,b,c} = alpha * A_{b,a,c}  + gamma * C_{a,b,c}
						        **********************/

					    std::vector<int> modeC{'a','b'};
					        std::vector<int> modeA{'a','b'};
						    int nmodeA = modeA.size();
						        int nmodeC = modeC.size();

							    std::unordered_map<int, int64_t> extent;
							        extent['a'] = hh;
								    extent['b'] = hh;

									    std::vector<int64_t> extentA;
									        for (auto mode : modeA)
											        extentA.push_back(extent[mode]);
										    std::vector<int64_t> extentC;
										        for (auto mode : modeC)
												        extentC.push_back(extent[mode]);

											    /**********************
											           * Allocating data
												        **********************/

											    size_t elementsA = 1;
											        for (auto mode : modeA)
													        elementsA *= extent[mode];
												    size_t elementsC = 1;
												        for (auto mode : modeC)
														        elementsC *= extent[mode];

													    size_t sizeA = sizeof(floatTypeA) * elementsA;
													        size_t sizeC = sizeof(floatTypeC) * elementsC;
														    printf("Total memory: %.2f GiB\n",(sizeA + sizeC)/1024./1024./1024);

														        void *A_d, *C_d, *D_d;
															    HANDLE_CUDA_ERROR(cudaMalloc((void**) &A_d, sizeA));
															        HANDLE_CUDA_ERROR(cudaMalloc((void**) &C_d, sizeC));
																    HANDLE_CUDA_ERROR(cudaMalloc((void**) &D_d, sizeC));

																        floatTypeA *A = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
																	    floatTypeC *C = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);

																	        if (A == NULL || C == NULL)
																			    {
																				            printf("Error: Host allocation of A or C.\n");
																					            return -1;
																						        }

																		    /*******************
																		           * Initialize data
																			        *******************/

																		    for(size_t i = 0; i < elementsA; i++){
																			            A[i] = (((float) rand())/RAND_MAX)*100;
																			}
																		        for(size_t i = 0; i < elementsC; i++){
																				        C[i] = (((float) rand())/RAND_MAX)*100;
																				}

																			    HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(C_d, sizeC, C, sizeC, sizeC, 1, cudaMemcpyDefault, 0));
																			        HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(D_d, sizeC, C, sizeC, sizeC, 1, cudaMemcpyDefault, 0));
																				    HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(A_d, sizeA, A, sizeA, sizeA, 1, cudaMemcpyDefault, 0));

																				        /*************************
																					       * Memcpy perf 
																					            *************************/

																				        double minTimeMEMCPY = 1e100;
																					    cudaDeviceSynchronize();
																					        GPUTimer timer;
																						    timer.start();
																						        HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(D_d, sizeC, C_d, sizeC, sizeC, 1, cudaMemcpyDefault, 0));
																							    cudaDeviceSynchronize();
																							        minTimeMEMCPY = timer.seconds();

																								    /*************************
																								           * cuTENSOR
																									        *************************/
																								    cutensorStatus_t err;
																								        cutensorHandle_t handle;
																									    HANDLE_ERROR(cutensorInit(&handle));

																									        /**********************
																										       * Create Tensor Descriptors
																										            **********************/
																									        cutensorTensorDescriptor_t descA;
																										    HANDLE_ERROR(cutensorInitTensorDescriptor( &handle,
																													                     &descA,
																															                      nmodeA,
																																	                       extentA.data(),
																																			                        NULL /* stride */,
																																						                 typeA, CUTENSOR_OP_IDENTITY));

																										        cutensorTensorDescriptor_t descC;
																											    HANDLE_ERROR(cutensorInitTensorDescriptor( &handle,
																														                     &descC,
																																                      nmodeC,
																																		                       extentC.data(),
																																				                        NULL /* stride */,
																																							                 typeC, CUTENSOR_OP_IDENTITY));

																											        double minTimeCUTENSOR = 1e100;
																												float time = 0.0;
																												    for (int i = 0; i < 20; i++)
																													        {
																															        HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(C_d, sizeC, C, sizeC, sizeC, 1, cudaMemcpyDefault, 0));
																																        HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
																																	        timer.start();
																																		        err = cutensorElementwiseBinary(&handle,
																																					                (void*)&alpha, A_d, &descA, modeA.data(),
																																							                (void*)&gamma, C_d, &descC, modeC.data(),
																																									                               C_d, &descC, modeC.data(),
																																												                       CUTENSOR_OP_MUL, typeCompute, 0 /* stream */);
																																			         time =time+ timer.seconds();
																																				        if (err != CUTENSOR_STATUS_SUCCESS)
																																						        {
																																								            printf("ERROR: %s\n", cutensorGetErrorString(err) );
																																									            }
																																					        minTimeCUTENSOR = (minTimeCUTENSOR < time)? minTimeCUTENSOR : time;
																																						    }

																						cout<<time/20<<endl;
																												        /*************************/

																												        double transferedBytes = sizeC;
																													    transferedBytes += ((float)alpha != 0.f) ? sizeA : 0;
																													        transferedBytes += ((float)gamma != 0.f) ? sizeC : 0;
																														    transferedBytes /= 1e9;
																														        printf("cuTensor: %.2f GB/s\n", transferedBytes / minTimeCUTENSOR);
																															    printf("memcpy: %.2f GB/s\n", 2 * sizeC / minTimeMEMCPY / 1e9 );

										HANDLE_CUDA_ERROR(cudaMemcpy2DAsync(C, sizeC, C_d, sizeA, sizeC, 1, cudaMemcpyDefault, 0));
																															        if (A) free(A);
																																    if (C) free(C);
																																        if (A_d) cudaFree(A_d);
																																	    if (C_d) cudaFree(C_d);
																																	        if (D_d) cudaFree(D_d);

	}																																		    return 0;
}
