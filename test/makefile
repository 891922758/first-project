link = -lcutensordecomposition -lcublas -lcusolver -lcurand -std=c++11
target = result
inc = -I ../inc/
libdir= -L ../lib/
obj = test_tucker4.cpp

$(target):$(obj)
	nvcc -o $@ $< $(libdir) $(inc) $(link)
.PHONY: clean
clean:
	rm $(target)
	
