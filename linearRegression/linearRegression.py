import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib.pyplot as plt
from autograd import grad
# Import Autograd modules here

#This is used to calculate rss value for all X and y values in the 2d space for a given theta.
X_in=np.ones(2)
y_in=np.ones(2)
def rss(a,b):
    return np.sum((y_in-a-b*X_in)**2)
# Here I wanted to use it as function of np vectors.

#This is for auto grad
def auto(x,y,t):
    return (y-x*t)**2
#This is each term of mse..    

class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
        pass

    def fit_non_vectorised(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        if(batch_size>X.shape[0]):
            print("Batch size has exceded the size of X")
            quit()
        self.coef_=np.zeros(X.shape[1]+1)
        X_arr=np.ones((1,X.shape[0]))
        if(not self.fit_intercept):
            X_arr=np.zeros((1,X.shape[0]))
        X_arr=np.append(X_arr,np.array(X).T,axis=0)
        X_arr=X_arr.T
        y_arr=np.array(y)
        if(lr_type=='constant'):
            iter=0
            bs=0
            be=bs+batch_size
            while(iter<n_iter):
                iter+=1
                X_arr_b=X_arr[bs:be]
                y_arr_b=y_arr[bs:be]
                coef=list(self.coef_)
                for cof in range(len(coef)):
                    size=X_arr_b.shape[0]
                    mse=0
                    for j in range(size):
                        mse+=(np.dot(X_arr_b[j],self.coef_)-y_arr_b[j])*X_arr_b[j][cof]
                    coef[cof]=self.coef_[cof]-lr*(mse/size)
                self.coef_=np.array(coef)    
                if(be>=X.shape[0]):
                    bs=0
                    be=bs+batch_size
                else:
                    bs=be
                    be=min(be+batch_size,X.shape[0])   

        elif(lr_type=='inverse'):
            iter=0
            bs=0
            be=bs+batch_size
            while(iter<n_iter):
                iter+=1
                X_arr_b=X_arr[bs:be]
                y_arr_b=y_arr[bs:be]
                coef=list(self.coef_)
                for cof in range(len(coef)):
                    size=X_arr_b.shape[0]
                    mse=0
                    for j in range(size):
                        mse+=(np.dot(X_arr_b[j],self.coef_)-y_arr_b[j])*X_arr_b[j][cof]
                    coef[cof]=self.coef_[cof]-(lr/iter)*(mse/size)
                self.coef_=np.array(coef)
                if(be>=X.shape[0]):
                    bs=0
                    be=bs+batch_size
                else:
                    bs=be
                    be=min(be+batch_size,X.shape[0])  
        else:
            print("Wrong lr_type given")  
            quit()

    def fit_vectorised(self, X, y,batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        if(batch_size>X.shape[0]):
            print("Batch size has exceded the size of X")
            quit()
        self.coef_=np.zeros(X.shape[1]+1)
        X_arr=np.ones((1,X.shape[0]))
        if(not self.fit_intercept):
            X_arr=np.zeros((1,X.shape[0]))
        X_arr=np.append(X_arr,np.array(X).T,axis=0)
        X_arr=X_arr.T
        y_arr=np.array(y)
        if(lr_type=='constant'):
            iter=0
            bs=0
            be=bs+batch_size
            while(iter<n_iter):
                iter+=1
                X_arr_b=X_arr[bs:be]
                y_arr_b=y_arr[bs:be]
                self.coef_=self.coef_-lr*np.matmul(X_arr_b.T,(np.matmul(X_arr_b,self.coef_)-y_arr_b))
                if(be>=X.shape[0]):
                    bs=0
                    be=bs+batch_size
                else:
                    bs=be
                    be=min(be+batch_size,X.shape[0])   

        elif(lr_type=='inverse'):
            iter=0
            bs=0
            be=bs+batch_size
            while(iter<n_iter):
                iter+=1
                X_arr_b=X_arr[bs:be]
                y_arr_b=y_arr[bs:be]
                self.coef_=self.coef_-(lr/iter)*np.matmul(X_arr_b.T,(np.matmul(X_arr_b,self.coef_)-y_arr_b))
                if(be>=X.shape[0]):
                    bs=0
                    be=bs+batch_size
                else:
                    bs=be
                    be=min(be+batch_size,X.shape[0])  
        else:
            print("Wrong lr_type given")        
            quit()
                
    def fit_autograd(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        # Here 2 indicated derivate with respect to 2nd variable.
        grad_mse=grad(auto,2)
        if(batch_size>X.shape[0]):
            print("Batch size has exceded the size of X")
            quit()
        self.coef_=np.zeros(X.shape[1]+1)
        X_arr=np.ones((1,X.shape[0]))
        if(not self.fit_intercept):
            X_arr=np.zeros((1,X.shape[0]))
        X_arr=np.append(X_arr,np.array(X).T,axis=0)
        X_arr=X_arr.T
        y_arr=np.array(y)
        if(lr_type=='constant'):
            iter=0
            bs=0
            be=bs+batch_size
            while(iter<n_iter):
                iter+=1
                X_arr_b=X_arr[bs:be]
                y_arr_b=y_arr[bs:be]
                coef=list(self.coef_)
                for cof in range(len(coef)):
                    size=X_arr_b.shape[0]
                    mse=0
                    for j in range(size):
                        x_temp=X_arr_b[j][cof]
                        t_temp=self.coef_[cof]
                        y_temp=y_arr_b[j]
                        for allcof in range(len(coef)):
                            if(allcof!=cof):
                                y_temp-=X_arr_b[j][allcof]*self.coef_[allcof]
                        mse+=grad_mse(x_temp,y_temp,t_temp)
                    coef[cof]=self.coef_[cof]-lr*(mse/size)
                self.coef_=np.array(coef)    
                if(be>=X.shape[0]):
                    bs=0
                    be=bs+batch_size
                else:
                    bs=be
                    be=min(be+batch_size,X.shape[0])   

        elif(lr_type=='inverse'):
            iter=0
            bs=0
            be=bs+batch_size
            while(iter<n_iter):
                iter+=1
                X_arr_b=X_arr[bs:be]
                y_arr_b=y_arr[bs:be]
                coef=list(self.coef_)
                for cof in range(len(coef)):
                    size=X_arr_b.shape[0]
                    mse=0
                    for j in range(size):
                        x_temp=X_arr_b[j][cof]
                        t_temp=self.coef_[cof]
                        y_temp=y_arr_b[j]
                        for allcof in range(len(coef)):
                            if(allcof!=cof):
                                y_temp-=X_arr_b[j][allcof]*self.coef_[allcof]
                        mse+=grad_mse(x_temp,y_temp,t_temp)       
                    coef[cof]=self.coef_[cof]-(lr/iter)*(mse/size)
                self.coef_=np.array(coef)
                if(be>=X.shape[0]):
                    bs=0
                    be=bs+batch_size
                else:
                    bs=be
                    be=min(be+batch_size,X.shape[0])  
        else:
            print("Wrong lr_type given") 
            quit()

    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''
        X_arr=np.ones((1,X.shape[0]))
        X_arr=np.append(X_arr,np.array(X).T,axis=0)
        X_arr=X_arr.T
        if(not self.fit_intercept):
            X_arr=np.array(X)
        y_arr=np.array(y)
        XTX=np.matmul(X_arr.T,X_arr)
        if(np.linalg.det(XTX)==0):
            print("Error, X transpose * X is not invertable")
            quit()
        else:
            self.coef_=np.matmul(np.linalg.inv(XTX),np.matmul(X_arr.T,y_arr))
        if(not self.fit_intercept):
            self.coef_=np.append(0,self.coef_)

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        X_arr=np.array(X)
        y_pred=np.matmul(X_arr,self.coef_[1:])
        y_pred=y_pred+self.coef_[0]
        return pd.Series(y_pred)

    def plot_surface(self, X, y, t_0, t_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """
        X_val=np.array(X)
        y_val=np.array(y)
        X_in=X_val
        y_in=y_val
        x_sur,y_sur=np.meshgrid(np.linspace(self.coef_[0]-5,self.coef_[0]+5,10),np.linspace(self.coef_[1]-5,self.coef_[1]+5,10))
        rss_z=np.vectorize(rss)
        z_sur=rss_z(x_sur,y_sur)
        fig=plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(x_sur, y_sur, z_sur,linewidth=0, antialiased=False)
        ax.scatter3D(t_0,t_1,rss_z(t_0,t_1)+20,color='red')
        ax.set_xlabel("Theta 0")
        ax.set_ylabel("Theta 1")
        ax.set_zlabel("RSS")
        plt.title("Surface Plot for RSS")
        plt.show()

    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """
        X_row=np.array(X)
        y_row=np.array(y)
        plt.scatter(X_row,y_row,color='red')
        X_line=np.linspace(np.min(X_row),np.max(X_row),5)
        y_line=X_line*t_1+t_0
        plt.plot(X_line,y_line)
        plt.xlim(np.min(X_row)-3,np.max(X_row)+3)
        plt.ylim(np.min(y_row)-3,np.max(y_row)+3)
        plt.xlabel("Features")
        plt.ylabel("Output")
        plt.title("Line fit plot")
        return plt

    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """
        X_val=np.array(X)
        y_val=np.array(y)
        X_in=X_val
        y_in=y_val
        x_sur,y_sur=np.meshgrid(np.linspace(-4,4,50),np.linspace(-4,4,50))
        rss_z=np.vectorize(rss)
        z_sur=rss_z(x_sur,y_sur)
        plt.contourf(x_sur, y_sur, z_sur)
        plt.scatter(t_0, t_1, rss(t_0,t_1)+5, color = "red")
        plt.title("Contour Plot for RSS")
        return plt
