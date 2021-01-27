cc = nvcc
prom = tensor
#source = tucker.cpp opera.cu kernel.cu
source = cp.cpp opera.cu kernel.cu
lib = -lcublas -lcusolver -std=c++11

$(prom):$(source)
	$(cc) -o $(prom) $(source) $(lib)

clean:
	rm -rf $(prom)
