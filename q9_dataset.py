from metrics import mae, rmse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from preprocessing.polynomial_features import PolynomialFeatures

N = 10
P = 2
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))
X[3]=4*X[1]

print(X)

# For Gradiant Method
model2=LinearRegression()
model2.fit_non_vectorised(X,y,10)
y_hat=model2.predict(X)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
print()

# For Normal Method
model=LinearRegression()
model.fit_normal(X,y)
y_hat=model.predict(X)
print(model.coef_)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
