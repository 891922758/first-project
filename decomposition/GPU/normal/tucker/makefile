cc=/usr/local/cuda-10.1/bin/nvcc
des=luhan
source = de_tucker4.cpp tucker4_hosvd.cu kernel.cu  
#source = input.cu hosvdtensorcore.cu kernel.cu  
#source = de_tucker.cpp tucker_hosvd.cu kernel.cu  
#source = de_tucker.cpp tucker_hooi.cu kernel.cu  
link = -lcublas -lcusolver -lcurand -lcufft -std=c++11

$(des):$(source)
	$(cc) -o $(des) $(source) $(link)

clean:
	rm -rf $(des)
