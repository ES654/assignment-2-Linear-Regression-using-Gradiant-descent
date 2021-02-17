from metrics import mae, rmse
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))
x=x.reshape(x.shape[0],1)
X_axis=list(range(1,11))
Y_axis=[0]*len(X_axis)
for i in range(len(X_axis)):
    mod=PolynomialFeatures(X_axis[i])
    x_t=mod.transform(x)
    model=LinearRegression()
    model.fit_non_vectorised(x_t,y,1,10,lr=1,lr_type='inverse')
    Y_axis[i]=np.log(np.abs(model.coef_).max())
plt.plot(X_axis,Y_axis)
plt.xlabel("Degree")
plt.ylabel("Max Theta (Log scale)")
plt.title("Max theta vs degree")
plt.show()
