import tensorly as tl
import numpy as np
from tensorly.decomposition import *
from tensorly import *
import time

def cp_als(tensor,r):
    a = tensor.shape[0]
    weights = tl.ones(r, **tl.context(tensor))
    factors = [np.random.random((tensor.shape[0], r)),np.random.random((tensor.shape[1],r)),np.random.random((tensor.shape[2],r))]
    for i in range(10):
        r1 = tl.dot(tl.conj(tl.transpose(factors[2])),factors[2])*tl.dot(tl.conj(tl.transpose(factors[1])), factors[1])
        l1 = unfolding_dot_khatri_rao(tensor, (weights, factors), 0)
        factors[0] = tl.transpose(tl.solve(tl.conj(tl.transpose(r1)), tl.transpose(l1)))

        r2 = tl.dot(tl.conj(tl.transpose(factors[2])),factors[2])*tl.dot(tl.conj(tl.transpose(factors[0])), factors[0])
        l2 = unfolding_dot_khatri_rao(tensor, (weights, factors), 1)
        factors[1] = tl.transpose(tl.solve(tl.conj(tl.transpose(r2)), tl.transpose(l2)))
        
        r3 = tl.dot(tl.conj(tl.transpose(factors[1])),factors[1])*tl.dot(tl.conj(tl.transpose(factors[0])), factors[0])
        l3 = unfolding_dot_khatri_rao(tensor, (weights, factors), 2)
        factors[2] = tl.transpose(tl.solve(tl.conj(tl.transpose(r3)), tl.transpose(l3)))

    return factors 

if __name__=='__main__':
    for i in range(50,2000,50):
        print(i)
        r = int(i*0.1)
        tensor= np.random.random((i,i,i))
        time_start=time.time()
        cp_als(tensor,r)
        time_end=time.time()
        print('time cost ',time_end-time_start,'s')
