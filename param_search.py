import numpy as np
import pandas as pd
import math
import sys
from scipy.special import softmax
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates

def loglikeihood(w,X,Y):
    r = X.shape[0]
    X = np.matmul(X,w)
    X = softmax(X,axis=1)
    ga = math.pow(10,-15)
    X = np.log(np.clip(X,ga,1-ga))
    Y = np.multiply(Y,X)
    ans = -np.sum(Y)/r
    return ans

trainfile = sys.argv[1]
testfile = sys.argv[2]
dftrain = pd.read_csv(trainfile,index_col = 0)
dftest = pd.read_csv(testfile,index_col = 0)
ytrain = dftrain['Length of Stay']
dftrain = dftrain.drop(columns = ['Length of Stay'])
cols = dftrain.columns
cols = cols[:-1]
dftrain = pd.get_dummies(dftrain, columns=cols, drop_first=True)
dftrain = dftrain.to_numpy()
r = dftrain.shape[0]
Xtrain = np.asarray(dftrain)
Xtrain = np.c_[np.ones(Xtrain.shape[0]),Xtrain]
r,c = Xtrain.shape
w = np.zeros((c, 8))
Xt = Xtrain.T
ytrain = np.asarray(pd.get_dummies(ytrain))
losses1 = []
losses2 = []
time = []
alpha = 0.45
beta = 0.75
k = int(math.sqrt(r))
i = 1
num = 100
n = 2.5
while (i <= num):
    X = np.matmul(Xtrain,w)
    yhat = softmax(X, axis=1)
    err = yhat-ytrain
    g = np.matmul(Xt,err)
    g = g/r
    fn = np.linalg.norm(g)
    fn = np.square(fn)
    fn = alpha*fn
    lx = loglikeihood(w,Xtrain,ytrain)
    while(loglikeihood(w-n*g,Xtrain,ytrain) > (lx-n*fn)):
        n = n*beta
    j = 0
    while((j+1)*k < r):
        X = np.matmul(Xtrain[j*k:(j+1)*k,:],w)
        yhat = np.asarray(softmax(X,axis=1))
        err = yhat-ytrain[j*k:(j+1)*k]
        g = np.matmul(Xtrain[j*k:(j+1)*k,:].T,err)
        g = g/k
        w = w-n*g
        j = j+1
    i = i+1
    losses1.append(loglikeihood(w,Xtrain,ytrain))
    losses2.append(np.linalg.norm(g))
    time.append(datetime.now())
dates = matplotlib.dates.date2num(time)
plt.plot_date(dates,losses1)
plt.xlabel("Time")
plt.ylabel("Loss")
plt.show()
