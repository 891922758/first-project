cc=/usr/local/cuda-10.1/bin/nvcc
des=luhan
source = de_cp.cpp cp_als.cu kernel.cu  
#source = input.cu tensorcore_cp.cu kernel.cu  
#source = de_cp4.cpp cp4_als.cu kernel.cu  
link = -lcublas -lcusolver -lcurand -lcufft -std=c++11

$(des):$(source)
	$(cc) -o $(des) $(source) $(link)

clean:
	rm -rf $(des)
