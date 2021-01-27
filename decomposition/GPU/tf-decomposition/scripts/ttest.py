import numpy as np
import tensorflow as tf
from scipy.io.matlab import loadmat
import time

import sys; sys.path.append('..')
from ttensor import TuckerTensor

#mat = loadmat('../data/bread/brod.mat')
#X = mat['X'].reshape([10,11,8])
for i in range(160,1300,160):
    print(i)
    X = np.random.random([i,i,i])
    r = np.int(i*0.1)
    T = TuckerTensor(X.shape, ranks=[r,r,r], regularize=0.0, init='random')
#X_predict = T.train_als(X, tf.train.AdadeltaOptimizer(0.05), epochs=15000)
#X_predict = T.hooi(X, epochs=100)
    start = time.clock()
    X_predict = T.hosvd(X)
    end = (time.clock()-start)
    f = open("hosvd.txt","a")
    f.write(str(i))
    f.write("  ")
    f.write(str(end))
    f.write('\n')
    f.close()
    print("time used",end)

#X_vec = X.reshape(100*100*100)
#X_predict_vec = X_predict.reshape(100*100*100)
#res = np.linalg.norm(X_vec-X_predict_vec)/np.linalg.norm(X_vec)
#print(res)

#np.save('../data/bread/X_tucker.npy', X_predict)
