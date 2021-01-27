cc=/usr/local/cuda-10.1/bin/nvcc
des=luhan
#source = de_cp.cpp cp_als.cu kernel.cu  
source = de_tucker.cpp tucker_hosvd.cu kernel.cu  
link = -lcublas -lcusolver -lcurand -std=c++11

$(des):$(source)
	$(cc) -o $(des) $(source) $(link)

clean:
	rm -rf $(des)
