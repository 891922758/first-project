import numpy as np
import time
from scipy.io.matlab import loadmat
import tensorflow as tf

import sys; sys.path.append('..')
from ktensor import KruskalTensor

#mat = loadmat('../data/bread/brod.mat')
#X = mat['X'].reshape([10,11,8])
for i in range(160,1300,160):
    print(i)
    r = np.int(i*0.1)
    X = np.random.random([i,i,i])
    T = KruskalTensor(X.shape, rank=r, regularize=1e-7, init='random', X_data=X)
    start = time.clock()
    X_predict = T.train_als(X, tf.train.AdadeltaOptimizer(0.05), epochs=10)
    end = (time.clock()-start)
    f = open("cp_als.txt","a")
    f.write(str(i))
    f.write("  ")
    f.write(str(end))
    f.write('\n')
    f.close()
    print("time used",end)

# T = KruskalTensor(X.shape, rank=3, regularize=0.0, init='nvecs', X_data=X)
# X_predict = T.train_als_early(X, tf.train.AdadeltaOptimizer(0.05), epochs=30000)

#X_vec = X.reshape(160*160*160)
#X_predict_vec = X_predict.reshape(160*160*160)
#res = np.linalg.norm(X_vec-X_predict_vec)/np.linalg.norm(X_vec)
print(res)
#np.save('./X_cp.npy', X_predict)
