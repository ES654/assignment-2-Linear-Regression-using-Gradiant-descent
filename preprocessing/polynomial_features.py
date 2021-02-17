''' In this file, you will utilize two parameters degree and include_bias.
    Reference https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
'''
import numpy as np
import itertools

class PolynomialFeatures():
    
    def __init__(self, degree=2,include_bias=True):
        """
        Inputs:
        param degree : (int) max degree of polynomial features
        param include_bias : (boolean) specifies wheter to include bias term in returned feature array.
        """
        self.degree=degree
        self.include_bias=include_bias
    
    def transform_1d(self,X_1):
        combinations=list(range(X_1.shape[0]))
        X_trans=np.array([1])
        if(self.include_bias):
            X_trans=np.append(X_trans,X_1,axis=0)
        else:
            X_trans=X_1
        for i in range(2,self.degree+1):
            for j in itertools.combinations_with_replacement(combinations,i):
                X_trans=np.append(X_trans,X_1[j[0]])
                for k in range(1,len(j)):
                    X_trans[-1]=X_trans[-1]*X_1[j[k]]
        return X_trans
    
    def transform(self,X):
        """
        Transform data to polynomial features
        Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. 
        For example, if an input sample is  np.array([a, b]), the degree-2 polynomial features with "include_bias=True" are [1, a, b, a^2, ab, b^2].
        
        Inputs:
        param X : (np.array) Dataset to be transformed
        
        Outputs:
        returns (np.array) Tranformed dataset.
        """
        X=np.array(X)
        if(X.ndim==1):
            return self.transform_1d(X)            
        elif(X.ndim==2):
            X_tran=self.transform_1d(X[0])
            for i in range(1,X.shape[0]):
                X_tran=np.vstack((X_tran,self.transform_1d(X[i])))
            return X_tran    
        else:
            print("Warning: The input array is not Transformed since its greater than 2 dimension")
            print("Its dimension is:{} required is 2".format(X.ndim))
            return X
