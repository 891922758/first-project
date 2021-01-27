
import tensorly as tl
import numpy as np
from tensorly.decomposition import *
from tensorly import *
import time

def hosvd(X,t_size):
    core, tucker_factors = tucker(X, ranks=t_size,n_iter_max = 1, init='svd', tol=10e-5)
    return core,tucker


if __name__=='__main__':
    for i in range(50,2000,50):
        print(i)
        r = int(i*0.1)
        t_size=[r,r,r]
        X= np.random.random((i,i,i))
        time_start=time.time()
        hosvd(X,t_size)
        time_end=time.time()
        f = open('tucker.txt','a')
        f.write(str(i))
        f.write('   ')
        f.write(str(time_end-time_start))
        f.write('\n')
        print('time cost ',time_end-time_start,'s')
