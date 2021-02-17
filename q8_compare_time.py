import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
from linearRegression.linearRegression import LinearRegression
from time import time


for i in [20,40,60]:
    N=10000
    P=i
    X=pd.DataFrame(np.random.randn(N,P))
    y=pd.Series(np.random.rand(N))
    model=LinearRegression()
    st=time()
    model.fit_normal(X,y)
    stp=time()
    print("Time for N={} and P={} is : {}".format(N,P,stp-st))

for i in [1000,10000,100000]:
    N=i
    P=50
    X=pd.DataFrame(np.random.randn(N,P))
    y=pd.Series(np.random.rand(N))
    model=LinearRegression()
    st=time()
    model.fit_normal(X,y)
    stp=time()
    print("Time for N={} and P={} is : {}".format(N,P,stp-st))

for i in [20,40,60,80,100]:
    N=10000
    P=50
    X=pd.DataFrame(np.random.randn(N,P))
    y=pd.Series(np.random.rand(N))
    model=LinearRegression()
    st=time()
    model.fit_vectorised(X,y,10000,i)
    stp=time()
    print("Time for N={} , P={} , iteration={} is : {}".format(N,P,i,stp-st))


N=1000
P=20
X=pd.DataFrame(np.random.randn(N,P))
y=pd.Series(np.random.rand(N))
model=LinearRegression()
st=time()
model.fit_normal(X,y)
stp=time()
y_hat=model.predict(X)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
print(stp-st)
