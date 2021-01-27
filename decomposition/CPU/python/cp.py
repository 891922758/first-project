import tensorly as tl
import numpy as np
from tensorly.decomposition import *
from tensorly import *
import time

def cp_als(X,r):
    factors = parafac(X, rank=r,n_iter_max=10,init = 'random',tol=10e-6)
   # res = tl.kruskal_to_tensor(factors)
   # res1 = res.reshape((a*a*a))
   # X1 = X.reshape((a*a*a))
   # error =( np.linalg.norm(res1-X1))/(np.linalg.norm(X1))
   # print(error)
    return factors 

if __name__=='__main__':
    for i in range(50,2000,50):
        print(i)
        r = int(i*0.1)
        X= np.random.random((i,i,i))
        time_start=time.time()
        cp_als(X,r)
        time_end=time.time()
        f = open('cp.txt','a')
        f.write(str(i))
        f.write('   ')
        f.write(str(time_end-time_start))
        f.write('\n')
        print('time cost ',time_end-time_start,'s')
