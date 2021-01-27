
Our cutensor-CP/Tucker GPU library

Ubuntu 18.04, cuda 10.1, c++11

We provide hadmard product, tensor product, khatri-rao product, tensor contraction (TTM), matriced tensor times khatri-rao product (MTTKRP),tensor matricization operation. Also, third- fourth- order CP and Tucker tensor decomposition, CP and Tucker decomposition with tensor core are included.

Using guide

1 Download git and unzip to your system

2 make

3 In the test content,you can test these operations

tip: our library is cublas-like, you should use allocate GPU memory for it. 

Decomposition fold includes matlab, tensorLab-CPU, tensorLab-GPU, tensorD-GPU and our GPU-Baseline and our optimized version.
