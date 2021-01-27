cc=/usr/local/cuda-10.1/bin/nvcc
des=luhan
source = de_cp.cu cp_als.cu kernel.cu  
#source = de_cp4.cpp 4_cp_als.cu kernel.cu  
#source = de_tucker4.cpp 4_tucker_hosvd.cu kernel.cu  
#source = de_tucker.cpp tucker_hosvd.cu kernel.cu  
#source = de_tucker.cpp tucker_hooi.cu kernel.cu  
link = -lcublas -lcusolver -lcurand -lcufft -std=c++11

$(des):$(source)
	$(cc) -o $(des) $(source) $(link)

clean:
	rm -rf $(des)
