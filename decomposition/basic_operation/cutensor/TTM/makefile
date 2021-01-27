cc=/usr/local/cuda-10.1/bin/nvcc
des=luhan
source = contraction.cu  
#source = test_ttm.cpp ttm.cu 

link = -I/home/iiplab2/Downloads/libcutensor/include/ -L/home/iiplab2/Downloads/libcutensor/lib/10.1/  -lcutensor -lcublas -lcusolver -lcurand -lcufft -std=c++11

$(des):$(source)
	$(cc) -o $(des) $(source) $(link)

clean:
	rm -rf $(des)
