
import numpy as np
from numpy.lib.function_base import select
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

# This is for Non_vectorized
for fit_intercept in [True, False]:
    for type in ['inverse','constant']:
        for batch in [1,X.shape[0]//2,X.shape[0]]:
            LR = LinearRegression(fit_intercept=fit_intercept)
            LR.fit_non_vectorised(X, y,batch,lr_type=type) # here you can use fit_non_vectorised / fit_autograd methods
            y_hat = LR.predict(X)
            print("Fit_intercept : {} , type : {} , batch_size : {}".format(str(fit_intercept),type,batch))
            print('RMSE: ', round(rmse(y_hat, y),3),end=" ")
            print('MAE: ', round(mae(y_hat, y),3))
            print()

# This is for vectorized
for fit_intercept in [True, False]:
    for type in ['inverse','constant']:
        for batch in [1,X.shape[0]//2,X.shape[0]]:
            LR = LinearRegression(fit_intercept=fit_intercept)
            LR.fit_vectorised(X, y,batch,lr_type=type) # here you can use fit_non_vectorised / fit_autograd methods
            y_hat = LR.predict(X)
            print("Fit_intercept : {} , type : {} , batch_size : {}".format(str(fit_intercept),type,batch))
            print('RMSE: ', round(rmse(y_hat, y),3),end=" ")
            print('MAE: ', round(mae(y_hat, y),3))
            print()

#This is for auto grad
for fit_intercept in [True, False]:
    for type in ['inverse','constant']:
        for batch in [1,X.shape[0]//2,X.shape[0]]:
            LR = LinearRegression(fit_intercept=fit_intercept)
            LR.fit_autograd(X, y,batch,lr_type=type) # here you can use fit_non_vectorised / fit_autograd methods
            y_hat = LR.predict(X)
            print("Fit_intercept : {} , type : {} , batch_size : {}".format(str(fit_intercept),type,batch))
            print('RMSE: ', round(rmse(y_hat, y),3),end=" ")
            print('MAE: ', round(mae(y_hat, y),3))
            print()
