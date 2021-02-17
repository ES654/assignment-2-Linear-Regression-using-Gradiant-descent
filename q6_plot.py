import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression

for k in range(1,5):
    x = np.array([i*np.pi/180 for i in range(60,300,k)])
    np.random.seed(10)  #Setting seed for reproducibility
    y = 4*x + 7 + np.random.normal(0,3,len(x))
    x=x.reshape(x.shape[0],1)
    X_axis=[1,3,5,7,9]
    Y_axis=[0,0,0,0,0]
    for i in range(len(X_axis)):
        mod=PolynomialFeatures(X_axis[i])
        x_t=mod.transform(x)
        model=LinearRegression()
        model.fit_non_vectorised(x_t,y,1,10,lr=1,lr_type='inverse')
        Y_axis[i]=np.log(np.abs(model.coef_).max())
    plt.plot(X_axis,Y_axis)
    plt.xlabel("Degree")
    plt.ylabel("Max Theta (Log scale)")
    samp_text="Max theta vs degree for n="+str(x.shape[0])
    plt.title(samp_text)
    plt.show()    
