import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression

X = np.array([-4,-3,-4,-2,-2,-3,-2,-1,1,0,-3,1,1,2,3,2,3,4,2,1,2,3,4,2,2,1,4,2,3,4])
y = pd.Series([1,4,0,5,3,1,5,6,4,5,3,4,7,5,7,8,6,9,7,8,9,7,8,9,6,5,8,4,10,11])
X=X.reshape((X.shape[0],1))
model=LinearRegression()
itr=0
for i in range(1,150,5):
    model=LinearRegression()
    model.fit_non_vectorised(X,y,30,i)
    coff=model.coef_
    sam=model.plot_line_fit(X,y,coff[0],coff[1])
    samn='line/im'+str(itr)+'.png'
    sam.savefig(samn)
    sam.cla()
    itr+=1

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

itr=0
sam=plt
for i in range(1,300,10):
    model=LinearRegression()
    model.fit_non_vectorised(X,y,30,i)
    coff=model.coef_
    sam=model.plot_contour(X,y,coff[0],coff[1])
    samn='cont/im'+str(itr)+'.png'
    sam.savefig(samn)
    itr+=1

itr=0
sam=plt
for i in range(1,300,10):
    model=LinearRegression()
    model.fit_non_vectorised(X,y,30,i)
    coff=model.coef_
    sam=model.plot_surface(X,y,coff[0],coff[1])
    samn='sur/im'+str(itr)+'.png'
    sam.savefig(samn)
    itr+=1
